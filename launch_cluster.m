


%% list of file path used for the run


savepath = ['./', 'results' , '/'];

savename = [savepath,'ans_',num2str(imprunnbr)]

% Next parameters are imported from the CMDL
%

pars
imprunnbr


HOPIT_Main(pars)

save(savename, 'ans')


clear all;
disp('run completed');
quit

