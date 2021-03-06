% Program for Risk sensitive MLP..........................................

clear all
close all
clc

% Load the training data..................................................
Ntrain=load('xyz.tra');
[NTD,~] = size(Ntrain);

% Initialize the Algorithm Parameters.....................................
inp = 19;          % No. of input neurons
hid = 100 ;        % No. of hidden neurons
out = 7;            % No. of Output Neurons
lam = 1.e-02;       % Learning rate
epo = 1000;

% Initialize the weights..................................................
Wi = 0.001*(rand(hid,inp)*2.0-1.0);  % Input weights
Wo = 0.001*(rand(out,hid)*2.0-1.0);  % Output weights


% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    miscla = 0;
    for sa = 1 : NTD
        xx = Ntrain(sa,1:inp)';     % Current Sample
        tt = Ntrain(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        er = tt - Yo;               % Error
        Wo = Wo + lam * (er * Yh'); % update rule for output weight
        Wi = Wi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx';    %update for input weight
        sumerr = sumerr + sum(er.^2);
        ca = find(tt==1);           % actual class
        [~,cp] = max(Yo);           % Predicted class
        if ca~=cp 
            miscla = miscla + 1;
        end
    end
    disp([sumerr miscla])
%     save -ascii Wi.dat Wi;
%     save -ascii Wo.dat Wo;
end

% Validate the network.....................................................
conftra = zeros(out,out);
res_tra = zeros(NTD,2);
for sa = 1: NTD
        xx = Ntrain(sa,1:inp)';     % Current Sample
        tt = Ntrain(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        ca = find(tt==1);           % actual class
        [~,cp] = max(Yo);           % Predicted class
        conftra(ca,cp) = conftra(ca,cp) + 1;
        res_tra(sa,:) = [ca cp];
end
disp(conftra)

% Test the network.........................................................
NFeature=load('xyz.tes');
[NTD,~]=size(NFeature);
conftes = zeros(out,out);
res_tes = zeros(NTD,2);
for sa = 1: NTD
        xx = NFeature(sa,1:inp)';   % Current Sample
        ca = NFeature(sa,end);      % Actual class
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        [~,cp] = max(Yo);           % Predicted class
        conftes(ca,cp) = conftes(ca,cp) + 1;
        res_tes(sa,:) = [ca cp];
end
disp(conftes)
