from mnist_data.dataset_utils import train_sets, dataset_inputs
from models.models import CNN, MLP
from dashboard.dashboard import DashboardTrain

[X_train, Y_train_raw, 
 X_val_raw, Y_val_raw, 
 X_val_raw1, Y_val_raw1, X_val_raw2, Y_val_raw2, 
 Y_train_onehot, Y_val_onehot, Y_val_onehot1, Y_val_onehot2, 
 X_train_cnn, X_val_cnn1, X_val_cnn2] = train_sets 
train_imgs, val_imgs, _, _ = dataset_inputs
cnn_model, mlp_model = CNN(), MLP()
model = cnn_model       # training model

#training inputs
epochs = 2 ** 3         # range(2, 4)
batch_size = 2 ** 6     # range(5, 7)
learning_rate = 0.002   # range(0.001, 0.010)
drop_prob = 0.000       # range(0.00, 0.35)
l2_lambda = 0.001       # range(0.001, 0.005)

#optimizer
optimizer='sgd' # sgd | adam
beta1=0.9
beta2=0.999
eps=1e-8

#outputs
batches_per_epoch = (train_imgs / batch_size)

#train model
model.printout(epochs, batches_per_epoch, train_imgs, val_imgs)

if model.type == "cnn":
  model.train_validate(X_train_cnn, Y_train_onehot, X_val_cnn1, Y_val_onehot1,
                          epochs=epochs, optimizer=optimizer, lr=learning_rate, batch_size=batch_size, l2_lambda=l2_lambda,
                          beta1=beta1, beta2=beta2, eps=eps)
elif model.type == "mlp":
  model.train_validate(X_train, Y_train_onehot, X_val_raw, Y_val_onehot,
                          epochs=epochs, optimizer=optimizer, lr=learning_rate, batch_size=batch_size, l2_lambda=l2_lambda,
                          beta1=beta1, beta2=beta2, eps=eps)
  #model.train_validate_2(epochs, trainer, X_train, Y_train_onehot, X_val_raw1, Y_val_onehot1, X_val_raw2, Y_val_onehot2)
else:
  print(f'{model.type} not available')   

#save model weights
model.save(f'weights\weights_{model.type}\e{epochs}bpe{batches_per_epoch}traimgs{train_imgs}valimgs{val_imgs}_weights_{optimizer}.npz')

#graphs
dashboard = DashboardTrain(model)
dashboard.plot_training_curves()
dashboard.plot_all_overlays()
