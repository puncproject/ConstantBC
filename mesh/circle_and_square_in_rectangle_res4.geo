res = 0.0375;
resc = 0.0125;
ress = 0.0125;
R = 0.15;
d = 0.3;
H = 0.4;
W = 0.7;
Point(1) = {-W, -H, 0, res};
Point(2) = {-W, H, 0, res};
Point(3) = {W, H, 0, res};
Point(4) = {W, -H, 0, res};
Point(5) = {-d, 0, 0, resc};
Point(6) = {-d-R, 0, 0, resc};
Point(7) = {-d+R, 0, 0, resc};
Point(8) = {-d, R, 0, resc};
Point(9) = {-d, -R, 0, resc};
Point(10) = {d-R, -R, 0, ress};
Point(11) = {d+R, -R, 0, ress};
Point(12) = {d+R, R, 0, ress};
Point(13) = {d-R, R, 0, ress};
Line(1) = {2, 1};
Line(2) = {1, 4};
Line(3) = {4, 3};
Line(4) = {3, 2};
Line(5) = {13, 10};
Line(6) = {10, 11};
Line(7) = {11, 12};
Line(8) = {12, 13};
Circle(9) = {6, 5, 9};
Circle(10) = {9, 5, 7};
Circle(11) = {7, 5, 8};
Circle(12) = {8, 5, 6};
Line Loop(13) = {1, 2, 3, 4};
Line Loop(14) = {12, 9, 10, 11};
Line Loop(15) = {5, 6, 7, 8};
Plane Surface(16) = {13, 14, 15};
Physical Line(17) = {1, 4, 3, 2};
Physical Line(18) = {9, 12, 11, 10};
Physical Line(19) = {5, 8, 7, 6};
Physical Surface(20) = {16};
