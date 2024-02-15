% Mihnea Mitrache
% Matlab script for drawing the graphs

% Execution times 10/2 samples per pixel
NUM_THREADS = [2 4 8 10 16];
SERIAL_10 = [285.776 285.776 285.776 285.776 285.776];
OPENMP_10 = [125.341 100.722 86.447 84.912 84.982];
CUDA_10 = [7.548 7.548 7.548 7.548 7.548];
MPI_10 = [165.533 111.118 102.474 101.912 102.782];
PTHREADS_10 = [115.031 89.513 77.912 76.912 76.986];
HYBRID_10 = [132.193 118.945 104.474 103.912 104.772];

SERIAL_2 = [77.095 77.095 77.095 77.095 77.095];
OPENMP_2 = [52.706 29.543 19.033 17.691 17.378];
CUDA_2 = [4.446 4.446 4.446 4.446 4.446];
MPI_2 = [52.706 31.832 19.360 18.140 18.120];
PTHREADS_2 = [38.768 19.471 18.140 16.768 16.511];
HYBRID_10 = [51.135 25.567 19.135 18.135 18.147];


% 1. Execution time - CPU Parallel Solutions
figure(1);
plot(NUM_THREADS, OPENMP, 'b', NUM_THREADS, MPI, 'c', NUM_THREADS, PTHREADS, 'm', NUM_THREADS, HYBRID, 'y', LineWidth=2);
title('Execution time CPU Parallel');
xlabel('Number of threads');
ylabel('Time (s)');
legend('OPENMP', 'MPI', 'PTHREADS', 'HYBRID');
print -dpng 1.png

% 2. Execution time - SERIAL, Best CPU Parallel, CUDA Bar Plot
figure(2);
BEST_TIMES = [285.776 76.912 7.548];
b = bar(BEST_TIMES, 'FaceColor', 'flat');
b.CData(1,:) = [0 0 1];
b.CData(2,:) = [1 0 0];
b.CData(3,:) = [0 1 0];
title('Execution time SERIAL, Best CPU Parallel, CUDA');
ylabel('Time (s)');
set(gca, 'XTickLabel', {'SERIAL', 'PTHREADS', 'CUDA'});
text(1:length(BEST_TIMES),BEST_TIMES/2 - 4,num2str(BEST_TIMES'),'vert','bottom','horiz','center');
print -dpng 2.png

% 3. Speedup - Parallel CPU Solutions Bar Plot
figure(3);
SPEEDUP = SERIAL ./ [OPENMP; MPI; PTHREADS; HYBRID];
b = bar(SPEEDUP', 'FaceColor', 'flat');
title('Speedup Parallel CPU Solutions');
xlabel('Number of threads');
ylabel('Speedup');
set(gca, 'XTickLabel', {'2', '4', '8', '10', '16'});
legend('OPENMP', 'MPI', 'PTHREADS', 'HYBRID');
print -dpng 3.png

% 4. Speedup - Parallel CPU Solutions Line Plot
figure(4);
title('Speedup Parallel CPU Solutions');
SPEEDUP = SERIAL ./ [OPENMP; MPI; PTHREADS; HYBRID];
plot(NUM_THREADS, SPEEDUP, LineWidth=2);
xlabel('Number of threads');
ylabel('Speedup');
legend('OPENMP', 'MPI', 'PTHREADS', 'HYBRID');
print -dpng 4.png

% 5. Speedup - Best CPU Parallel, CUDA Bar Plot
figure(5);
PTHREAD_BEST = 285.776 / 76.912;
CUDA_BEST = 285.776 / 7.548;
SPEEDUP = [PTHREAD_BEST CUDA_BEST];
b = bar(SPEEDUP, 'FaceColor', 'flat');
b.CData(1,:) = [0 0 1];
b.CData(2,:) = [1 0 0];
title('Speedup Best CPU Parallel, CUDA');
ylabel('Speedup');
set(gca, 'XTickLabel', {'PTHREADS', 'CUDA'});
text(1:length(SPEEDUP),SPEEDUP/2 - 1,num2str(SPEEDUP'),'vert','bottom','horiz','center');
print -dpng 5.png

