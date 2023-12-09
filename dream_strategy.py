import torch
from fast_pytorch_kmeans import KMeans
class NEW_Strategy:
    def __init__(self, images, net, device):
        self.images = images
        self.net = net
        self.device = device

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

        index = torch.arange(len(embeddings), device=self.device)

        kmeans = KMeans(n_clusters=n, mode='euclidean')
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids

        dist_matrix = self.euclidean_dist(centers, embeddings)
        q_idxs = index[torch.argmin(dist_matrix,dim=1)]
        return q_idxs
    
    def get_embeddings(self, images):
        
        features = []
        embed = self.net.module.embed
        print(images[0])
        for i in range(len(images)):
            image = images[i].to(self.device)
            print(self.device)
            print(image.device)
            print(image.size())
            with torch.no_grad():
                features.append(embed(torch.unsqueeze(image, dim=0)).detach())
        features = torch.cat(features, dim=0)

        return features.to(self.device)

