
function [w,c,k] = perceptron3(training_data, T)
    [row,col] = size(training_data);
    w = cell(1, 10);
    c = cell(1, 10);
    k = [];
    x = training_data(:,1:col-1);
    y = training_data(:,col);
    tic
    for label = 1:10
        wk = [];
        ck = [];
        w0 = zeros(1,col-1);
        kk = 1;
        c1 = 1;
        wk = [wk; w0];
        
        ck = [ck; c1];
        for iteration = 1:T % # of iteration
            i = mod(iteration,row)+1;
            if y(i) == label-1
                new_label = 1;
            else
                new_label = -1;
            end
            
            if  dot(wk(kk,:),x(i,:)) * new_label <= 0
                w1 = wk(kk,:) + new_label * x(i,:);
                wk = [wk; w1];               
                ck = [ck; 1];
                kk = kk + 1;
            else
                ck(kk) = ck(kk) + 1;
            end
        end
        
        w{label} = wk;
        c{label} = ck;
        k = [k; kk];
    end
    toc
end    