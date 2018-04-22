// Resolution on outer and inner boundaries
reso = 0.0375;
resi = reso/3;

// Geometry
r = 0.15; // In-radius of inner boundaries (objects)
d = 0.3;  // Half-distance between object centers
H = 0.4;  // Half-height of bounding box
W = 0.7;  // Half-width of bounding box

// Coefficients for square and pentagon
// (http://mathworld.wolfram.com/Pentagon.html)
R  = 1.236*r; // Circum-radius
c1 = 0.309*R;
c2 = 0.809*R;
s1 = 0.951*R;
s2 = 0.587*R;

// Outer boundary
Point(1) = {-W, -H, 0, reso};
Point(2) = {-W,  H, 0, reso};
Point(3) = { W,  H, 0, reso};
Point(4) = { W, -H, 0, reso};

// Pentagon
Point(5) = {-d   ,  R , 0, resi};
Point(6) = {-d+s1,  c1, 0, resi};
Point(7) = {-d+s2, -c2, 0, resi};
Point(8) = {-d-s2, -c2, 0, resi};
Point(9) = {-d-s1,  c1, 0, resi};

// Square
Point(10) = {d-r, -r, 0, resi};
Point(11) = {d+r, -r, 0, resi};
Point(12) = {d+r,  r, 0, resi};
Point(13) = {d-r,  r, 0, resi};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 5};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 10};
Physical Line(1) = {1, 2, 3, 4};
Physical Line(2) = {5, 6, 7, 8, 9};
Physical Line(3) = {10, 11, 12, 13};
Line Loop(1) = {2, 3, 4, 1};
Line Loop(2) = {9, 5, 6, 7, 8};
Line Loop(3) = {12, 13, 10, 11};
Plane Surface(1) = {1, 2, 3};
Physical Surface(4) = {1};
