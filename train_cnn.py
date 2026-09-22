from mnist_data.dataset_utils import train_sets_cnn, dataset_inputs
from models.models import CNN
from dashboard.dashboard import DashboardTrain

X_train_cnn, Y_train_onehot, X_val_cnn1, Y_val_onehot1 = train_sets_cnn 
train_imgs, val_imgs, _, _ = dataset_inputs
model = CNN() # training model

#training inputs
epochs = 2 ** 3         # range(2, 4)
batch_size = 2 ** 6     # range(5, 9) number of images per update
learning_rate = 0.001   # range(0.001, 0.010)
drop_prob = 0.000       # range(0.00, 0.35)
l2_lambda = 0.003       # range(0.001, 0.005)

#optimizer inputs
optimizer='sgd' # sgd | adam
beta1=0.9
beta2=0.999
eps=1e-8

#outputs
batches_per_epoch = (train_imgs / batch_size)

#train model
model.printout(epochs, batches_per_epoch, train_imgs, val_imgs)

model.train_validate(X_train_cnn, Y_train_onehot, X_val_cnn1, Y_val_onehot1,
                     epochs=epochs, optimizer=optimizer, lr=learning_rate, batch_size=batch_size, l2_lambda=l2_lambda,
                     beta1=beta1, beta2=beta2, eps=eps)

#save model weights
model.save(f'weights\weights_{model.type}'
           f'\e{epochs}\{batches_per_epoch}bpe'
           f'_lr-{learning_rate}_{optimizer}'
           f'_t-{train_imgs}_v-{val_imgs}.npz')

#graphs
dashboard = DashboardTrain(model)
dashboard.plot_training_curves()
dashboard.plot_all_overlays()
