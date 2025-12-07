n_motors = 8;
motor_angles = zeros(1, n_motors);
for i = 1:n_motors
        motor_angles(i) = rand()*50-25;
end
tail_centre_angle =  rand()*180-90;
tic;
[xpos,ypos,caudal_fin,r_theta] = Kinematics_Fish(motor_angles',tail_centre_angle);
t0 = toc;
fprintf("Time Kinematics Fish: %f\n", t0)