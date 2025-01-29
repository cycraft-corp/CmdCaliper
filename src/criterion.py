import torch
import torch.nn.functional as F

class InfoNCE:
    def __init__(self, criterion_args, device="cuda"):
        self.device = device

        self.temperature = criterion_args.temperature

    def __call__(self, x, auxiliary_data):
        step_size = 3 if auxiliary_data["has_negative_sample"] else 2
        query_x = x[0::step_size]
        positive_x = x[1::step_size]
        if auxiliary_data["has_negative_sample"]:
            negative_x = x[2::step_size]

        positive_similarity = F.cosine_similarity(query_x, positive_x).unsqueeze(-1)
        positive_negative_similarity = F.cosine_similarity(
        query_x.unsqueeze(0), positive_x.unsqueeze(1), -1
        )
        label_mask = ~torch.eye(positive_negative_similarity.shape[0], device=self.device, dtype=torch.bool)
        positive_negative_similarity = positive_negative_similarity[label_mask].reshape(query_x.size(0), -1)
        if auxiliary_data["has_negative_sample"]:
            negative_similarity = F.cosine_similarity(
                query_x.unsqueeze(0), negative_x.unsqueeze(1), -1
            )
            positive_negative_similarity = torch.cat([positive_negative_similarity, negative_similarity], -1)
        all_similarity = torch.cat([positive_similarity, positive_negative_similarity], -1)
        labels = torch.zeros(all_similarity.size(0), dtype=torch.long, device=self.device)

        loss = F.cross_entropy(all_similarity / self.temperature, labels)
        loss = loss.mean()
        return loss
