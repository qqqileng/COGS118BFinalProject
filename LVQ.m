data = table2array(hmnist2828RGB);
X_train = table2array(hmnist2828RGB(1:4000, 1:2352));
Y_train = table2array(hmnist2828RGB(1:4000, 2353));
X_test = table2array(hmnist2828RGB(4000:5000, 1:2352));
Y_test = table2array(hmnist2828RGB(4000:5000, 2353));
Y_train = ind2vec(Y_train');
net = lvqnet(8);
net.trainParam.epochs = 50;
net = train(net,X_train',Y_train);
view(net)
Y_pred = net(X_test');
perf = perform(net,Y_pred,Y_test)
classes = vec2ind(Y_pred);