import torch
import torch.nn as nn
import torch.optim as optim

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(FeatureExtractor, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.class_specific = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim)
            ) for _ in range(num_classes)
        ])
        self.attention = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        shared_features = self.shared(x)
        attention_weights = self.attention(shared_features)
        class_features = [f(shared_features) for f in self.class_specific]
        combined_features = sum([w[:, i].unsqueeze(1) * f for i, f in enumerate(class_features)])
        return combined_features, attention_weights

class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classes)
        self.thresholds = nn.Parameter(torch.zeros(num_classes))

    def forward(self, z):
        logits = self.fc(z)
        return logits - self.thresholds

class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.fc(z)

class IADA(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(IADA, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, latent_dim, num_classes)
        self.classifier = Classifier(latent_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(latent_dim)

    def forward(self, x):
        z, attention_weights = self.feature_extractor(x)
        class_output = self.classifier(z)
        domain_output = self.domain_discriminator(z)
        return class_output, domain_output, attention_weights

def train_iada(model, source_loader, target_loader, num_epochs, lr, lambda_adv, lambda_reg):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cls_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        model.train()
        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
            # Source domain
            s_class_output, s_domain_output, _ = model(source_data)
            cls_loss = cls_criterion(s_class_output, source_labels)
            
            # Target domain
            _, t_domain_output, _ = model(target_data)
            
            # Adversarial loss
            s_domain_labels = torch.ones(source_data.size(0), 1)
            t_domain_labels = torch.zeros(target_data.size(0), 1)
            adv_loss = adv_criterion(s_domain_output, s_domain_labels) + \
                       adv_criterion(t_domain_output, t_domain_labels)
            
            # Regularization
            reg_loss = torch.norm(model.classifier.thresholds, p=2)
            
            # Total loss
            loss = cls_loss - lambda_adv * adv_loss + lambda_reg * reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Usage example
input_dim = 100  # Replace with actual input dimension
latent_dim = 64
num_classes = 10
model = IADA(input_dim, latent_dim, num_classes)

# Assume source_loader and target_loader are defined
train_iada(model, source_loader, target_loader, num_epochs=100, lr=0.001, lambda_adv=0.1, lambda_reg=0.01)
