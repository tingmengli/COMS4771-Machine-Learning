%construct perceptron w0 ~ w9, each gives a confidence number
function [w, y_labels] = kernel_perceptron1(training_data,T)
    [row, col] = size(training_data);
    w = cell(1, 10);
    y_labels = cell(1, 10);
    x = training_data(:,1:col-1);
    y = training_data(:,col);
    
    for label = 1:10
        y_label = zeros(row, 1);
        for n = 1:row
            if y(n) == label-1
                y_label(n) = 1;
            else
                y_label(n) = -1;
            end
        end
        y_labels{label} = y_label;
        
        alpha = zeros(row, 1);
        for iteration = 1:T % # of iteration
            fprintf("label %d, iteration: %d\n", label-1, iteration);
            i = mod(iteration,row)+1;
            sum1 = 0;
            for r = 1:row
                d = dot(x(r,:), x(i,:));
                sum1 = sum1 + alpha(r)*y_label(r)*(1+d)^5;
            end
            if sum1 * y_label(i) <= 0
                alpha(i) = alpha(i) + 1;
            end
        end
        w{label} = alpha;
    end
    
end    