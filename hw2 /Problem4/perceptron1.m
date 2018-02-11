%construct perceptron w0 ~ w9, each gives a confidence number
function w = perceptron1(training_data, T)
    [row,col] = size(training_data);
    w = cell(1, 10);
    x = training_data(:,1:col-1);
    y = training_data(:,col);
    
    for label = 1:10
        w0 = zeros(1,col-1);
        for iteration = 1:T % # of iteration
            
            i = mod(iteration,row)+1;
            if y(i) == label-1
                if dot(w0,x(i,:)) <= 0
                    w0 = w0 + x(i,:);
                end
            elseif y(i) ~= label-1
                if dot(w0,x(i,:)) >= 0
                    w0 = w0 + (-1) .* x(i,:);
                end
            end
        end
    w{label} = w0;
    end
    
end    