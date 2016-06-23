function net = cnngradcheck(net, x, y)
epsilon = 1e-4;
er = 1e-6;
disp('performing numerical gradient checking...');

% Output bias
% for i = 1 : size(net.o,1)
%     p_net = net; p_net.ffb(i) = p_net.ffb(i) + epsilon;
%     m_net = net; m_net.ffb(i) = m_net.ffb(i) - epsilon;
%     
%     [m_net, p_net] = netrun(m_net, p_net, x, y);
%     d = (p_net.L - m_net.L) / (2 * epsilon);
%     
%     e = abs(d - net.dffb(i));
%     if e > er
%         disp('OUTPUT BIAS numerical gradient checking failed');
%         disp(e);
%         disp(d / net.dffb(i));
%         keyboard
%     end
% end
% Output weights
% for i = 1 : size(net.ffW,1)
%     for j = 1 : size(net.ffW,2)
% 
%         p_net = net; p_net.ffW(i,j) = p_net.ffW(i,j) + epsilon;
%         m_net = net; m_net.ffW(i,j) = m_net.ffW(i,j) - epsilon;
% 
%         [m_net, p_net] = netrun(m_net, p_net, x, y);
%         d = (p_net.L - m_net.L) / (2 * epsilon);
% 
%         e = abs(d - net.dffW(i,j))
%         if e > er
%             disp('OUTPUT WEIGHTS numerical gradient checking failed');
%             disp(e);
%             disp(d / net.dffW(i,j));
%             keyboard
%         end
%     end
% end
% Weights and bias in hidden layers
for l = 2 : numel(net.layers)
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : numel(net.layers{l}.a)
            
            for ii = 1 : numel(net.layers{l - 1}.a)
                
                for k = 1 : length(net.layers{l}.k{ii}{j})
                    p_net = net; p_net.layers{l}.k{ii}{j}(k) = p_net.layers{l}.k{ii}{j}(k) + epsilon;
                    m_net = net; m_net.layers{l}.k{ii}{j}(k) = m_net.layers{l}.k{ii}{j}(k) - epsilon;
                    
                    [m_net, p_net] = netrun(m_net, p_net, x, y);
                    d = (p_net.L - m_net.L) / (2 * epsilon);
                    
                    e = abs(d - net.layers{l}.dk{ii}{j}(k))
                    if e > er
                        disp('Hidden weights numerical gradient checking failed');
                        disp(e);
                        disp(d / net.layers{l}.dk{ii}{j}(k));
                        keyboard
                    end
                end
                
            end
            
%             for k = 1 : length(net.layers{l}.b{j})
%                 p_net = net; p_net.layers{l}.b{j}(k) = p_net.layers{l}.b{j}(k) + epsilon;
%                 m_net = net; m_net.layers{l}.b{j}(k) = m_net.layers{l}.b{j}(k) - epsilon;
%                 
%                 [m_net, p_net] = netrun(m_net, p_net, x, y);
%                 d = (p_net.L - m_net.L) / (2 * epsilon);
%                 
%                 e = abs(d - net.layers{l}.db{j}(k));
%                 if e > er
%                     disp('Hidden bias numerical gradient checking failed');
%                     disp(e);
%                     disp(d / net.layers{l}.db{j}(k));
%                     keyboard
%                 end
%             end
                        
        end
    end
end

disp('done')

end

function [m_net, p_net] = netrun(m_net, p_net, x, y)
m_net = cnnff(m_net, x); m_net = cnnbp(m_net, y);
p_net = cnnff(p_net, x); p_net = cnnbp(p_net, y);
end

