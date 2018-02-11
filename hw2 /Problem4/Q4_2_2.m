A = load('hw1data.mat');
x = A.X;
y = A.Y;

%split training and testing data in diff sizes
error_rates1 = zeros(1, 5);

Ts = [2000, 4000, 6000, 8000, 10000];
for s = 1:5
    t = Ts(s);
    training_x = x(1:8000,:);
    test_x = x(8000+1:10000,:);
    training_y = y(1:8000);
    test_y = y(8000+1:10000);
    training_data=[];
    training_data=cat(2,training_data,training_x,training_y);

    %train perceptron
    w2 = perceptron2(training_data, t);

    %test perceptron2
    test_err = 0;
    test_size = 10000-8000;
    result = zeros(1,10);
    for i = 1:2000
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
plot(Ts,error_rates1);

axis([1800 10200 0 0.5]);
