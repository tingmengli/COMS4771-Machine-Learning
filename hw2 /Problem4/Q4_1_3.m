A = load('hw1data.mat');

i = randperm(10000);
x = A.X(i,:);
y = A.Y(i);

%split training and testing data in diff sizes
error_rates = zeros(1, 5);
train_sizes = [7500, 8000, 8500, 9000, 9500];
for s = 1:5
    size = train_sizes(s);
    training_x = x(1:size,:);
    test_x = x(size+1:10000,:);
    training_y = y(1:size);
    test_y = y(size+1:10000);
    training_data=cat(2,training_x,training_y);

    %train perceptron
    [w3, c3, k3] = perceptron3(training_data, 2000);

    %test perceptron3
    test_err = 0;

    for i = 1:10000-size
        result = zeros(1,10);
        for label = 1:10
            w = w3{label};
            c = c3{label};
            k = k3(label);

            for j = 1:k
                result(label) = result(label)+ c(j) * sign(dot(w(j,:),test_x(i,:))); 
            end        
        end
        [max_conf,idx] = max(result);
        if idx-1 ~= test_y(i)
            test_err = test_err+1;
            %fprintf("test %d, error rate: %.3f\n", i, test_err/i);
        end
    end
    error_rate = test_err/(10000-size);
    fprintf("perceptron3 error rate: %.3f\n", error_rate);
    error_rates(s) = error_rate;
end

figure
plot(train_sizes,error_rates);
axis([7400 9600 0 0.5]);

