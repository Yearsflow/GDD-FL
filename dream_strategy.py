import torch
from fast_pytorch_kmeans import KMeans
class NEW_Strategy:
    def __init__(self, images, net, dataset):
        self.images = images
        self.net = net
        self.dataset = dataset

    def euclidean_dist(self,x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def query(self, n):

        embeddings = self.get_embeddings(self.images)

        index = torch.arange(len(embeddings),device='cuda')

        kmeans = KMeans(n_clusters=n, mode='euclidean')
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids

        dist_matrix = self.euclidean_dist(centers, embeddings)
        q_idxs = index[torch.argmin(dist_matrix,dim=1)]
        return q_idxs
        
    def query_match(self, n, sub_sample):
        k = n//sub_sample

        embeddings = self.get_embeddings(self.images)

        kmeans = KMeans(n_clusters=k, mode='euclidean')
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids

        dist_matrix = self.euclidean_dist(centers, embeddings)
        min_indices = torch.topk(dist_matrix, k=sub_sample, largest=False, dim=1)[1]

        q_idxs = min_indices.view(-1)
        return q_idxs
        
    def query_tiny(self, n):

        embeddings = self.get_embeddings(self.images)

        kmeans = KMeans(n_clusters=n, mode='euclidean')
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids

        dist_matrix = self.euclidean_dist(embeddings,centers)
        min_distance = torch.min(dist_matrix,dim=1)[0]
        min_distance_index = torch.argsort(min_distance)[:n]

        return min_distance_index
    
    def get_embeddings(self, images):

        if self.dataset in ['mnist', 'cifar10']:
            embed = self.net.module.embed
            with torch.no_grad():
                features = embed(images).detach()
        else:
            embed = self.net.module.embed
            features = []
            batch_size = 64
            batch_idx = 0
            with torch.no_grad():
                while True:
                    if batch_idx * batch_size >= len(images):
                        break
                    features.append(embed(images[batch_idx * batch_size: min((batch_idx + 1) * batch_size, len(images))]).detach())
                    batch_idx += 1
            features = torch.cat(features, dim=0)
        return features

