A = load('hw1data.mat');

i = randperm(10000);
x = A.X(i,:);
y = A.Y(i);

%split training and testing data in diff sizes
error_rates = zeros(1, 5);
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
    [w3, c3, k3] = perceptron3(training_data, t);

    %test perceptron3
    test_err = 0;

    for i = 1:10000-8000
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
    error_rate = test_err/(10000-8000);
    fprintf("perceptron3 error rate: %.3f\n", error_rate);
    error_rates(s) = error_rate;
end

figure
plot(Ts,error_rates);
axis([1800 10200 0 0.5]);
