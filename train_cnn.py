from utils.dataset_utils import train_sets_cnn, dataset_inputs
from utils.optimizer import SGD, Adam
from models.models import CNN, Trainer
from dashboard.dashboard import DashboardTrain

X_train_cnn, Y_train_onehot, X_val_cnn1, Y_val_onehot1 = train_sets_cnn 
train_imgs, val_imgs, _, _ = dataset_inputs
model = CNN() # training model

#training inputs
epochs = 2 ** 2         # range(2, 4)
batch_size = 2 ** 5     # range(5, 9) number of images per update
learning_rate = 0.003   # range(0.001, 0.010)
l2_lambda = 0.003       # range(0.001, 0.005)

#optimizer inputs
beta1=0.9
beta2=0.999
eps=1e-8
optimizer = Adam(
  lr=learning_rate,
  beta1=beta1,
  beta2=beta2,
  eps=eps
)
optimizer = SGD(lr=learning_rate)

#outputs
batches_per_epoch = (train_imgs / batch_size)

#train model
trainer = Trainer(model=model, optimizer=optimizer, l2_lambda=l2_lambda)

trainer.printout(epochs, batches_per_epoch, train_imgs, val_imgs)
trainer.train_validate(X_train_cnn, Y_train_onehot, 
                       X_val_cnn1, Y_val_onehot1, 
                       epochs, batch_size)

#save model weights
model.save(f'weights\weights_{model.type}'
           f'\e{epochs}\_{batches_per_epoch}bpe'
           f'_lr-{learning_rate}_{trainer.optimizer.type}'
           f'_t-{train_imgs}_v-{val_imgs}.npz')

#graphs
dashboard = DashboardTrain(trainer)
dashboard.plot_training_curves()
dashboard.plot_all_overlays()
