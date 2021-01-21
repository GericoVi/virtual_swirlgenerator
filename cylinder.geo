
r = DefineNumber[ 0.5, Name "Parameters/r" ];
cos = DefineNumber[ 0.70710678, Name "Parameters/cos" ];

Point(1) = {0, 0, 0, 1.0};

Point(2) = {r/2*cos, r/2*cos, 0, 1.0};
Point(3) = {r/2*cos, -r/2*cos, 0, 1.0};
Point(4) = {-r/2*cos, -r/2*cos, 0, 1.0};
Point(5) = {-r/2*cos, r/2 *cos, 0, 1.0};

Point(6) = {r*cos, r*cos, 0, 1.0};
Point(7) = {r*cos, -r*cos, 0, 1.0};
Point(8) = {-r*cos, -r*cos, 0, 1.0};
Point(9) = {-r*cos, r*cos, 0, 1.0};

Line(1) = {2,3};
Line(2) = {3,4};
Line(3) = {4,5};
Line(4) = {5,2};

Circle(5) = {6,1,7};
Circle(6) = {7,1,8};
Circle(7) = {8,1,9};
Circle(8) = {9,1,6};

Line(9) = {2,6};
Line(10) = {3,7};
Line(11) = {4,8};
Line(12) = {5,9};

Curve Loop(1) = {1, 2, 3, 4};

Curve Loop(2) = {-1,9,5,-10};
Curve Loop(3) = {-2,10,6,-11};
Curve Loop(4) = {-3,11,7,-12};
Curve Loop(5) = {-4,12,8,-9};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};

Recombine Surface {1,2,3,4,5};

Transfinite Line {1,2,3,4,5,6,7,8} = 20 Using Progression 1;
Transfinite Line {9,10,11,12}	= 45 Using Progression 0.85;

Transfinite Surface {1,2,3,4,5};

Extrude {0,0,2} {
	Surface {1,2,3,4,5}; Layers {50}; Recombine;
}

Physical Surface ('inlet')  = {1,2,3,4,5};
Physical Surface ('outlet') = {34,56,78,100,122};
Physical Surface ('walls') = {51,73,95,117};

Physical Volume ('domain') = {1,2,3,4,5};
