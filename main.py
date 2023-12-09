from cnn_model import CNNModel

cnnModel = CNNModel()
cnnModel.load_images()
cnnModel.prepare_data()
cnnModel.build_model_layers()
cnnModel.train_model()
cnnModel.test_model()
