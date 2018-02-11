A = load('hw1data.mat');
i = randperm(10000);
x = A.X(i,:);
y = A.Y(i);

%split training and testing data in diff sizes
error_rates1 = zeros(1, 5);

train_sizes = [7500, 8000, 8500, 9000, 9500];
for s = 1:5
    size = train_sizes(s);
    training_x = x(1:size,:);
    test_x = x(size+1:10000,:);
    training_y = y(1:size);
    test_y = y(size+1:10000);
    training_data=cat(2,training_x,training_y);

    %train perceptron
    w2 = perceptron2(training_data, 2000);

    %test perceptron2
    test_err = 0;
    test_size = 10000-size;
    result = zeros(1,10);
    for i = 1:test_size
        for label = 1:10
            result(label) = dot(w2(label,:),test_x(i,:)); 
        [max_conf,idx] = max(result);
        end
        if idx-1 ~= test_y(i)
            test_err = test_err+1;
            %fprintf("test %d, error rate: %.3f\n", i, test_err./i);
        end
    end
    error_rate = test_err/test_size;
    fprintf("perceptron1 error rate: %.3f\n", error_rate);
    error_rates1(s) = error_rate;
end

figure
plot(train_sizes,error_rates1);

axis([7400 9600 0 0.5]);
