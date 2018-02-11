%construct perceptron w0 ~ w9, each gives a confidence number
function w = perceptron2(training_data, T)
    [row,col] = size(training_data);
    w = [];
    x = training_data(:,1:col-1);
    y = training_data(:,col);
    
    tic
    for label = 1:10
        terminate = 0;
        while terminate == 0
            w0 = zeros(1,col-1);
            for iteration = 1:T % # of iteration
            %fprintf("iteration: %d\n", iteration);
                i = zeros(1, row);
                for j = 1:row
                    if y(j) == label-1
                        i(j) = dot(w0, x(j,:));
                    else
                        i(j) = dot(-1 * w0, x(j,:));
                    end
                end
                [val,idx] = min(i);

                if y(idx) == label-1
                    if dot(w0,x(idx,:)) <= 0
                        w0 = w0 + x(idx,:);
                    else
                        w = cat(1,w,w0);
                        disp(terminate);
                        break;
                    end
                elseif y(idx) ~= label-1
                    if (-1) * dot(w0,x(idx,:)) <= 0
                        w0 = w0 + (-1) * x(idx,:);
                    else
                        w = cat(1,w,w0);
                        disp(terminate);
                        break;
                    end
                end

            end
            w = cat(1,w,w0);
            break;
        end
    end
    toc
end    