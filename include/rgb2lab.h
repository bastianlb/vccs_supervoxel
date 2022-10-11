#include <inttypes.h>
#include <stdio.h>
#include <math.h>

struct RGB {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct LAB {
  double l;
  double a;
  double b;
};

double lab_dist(LAB l1, LAB l2) {
  return sqrt(
          pow(l1.l - l2.l, 2) +
          pow(l1.a - l2.a, 2) +
          pow(l1.b - l2.b, 2)
        );
}

double H(double q)
{
	double value;
	if ( q > 0.008856 ) {
		value = pow ( q, 0.333333 );
		return value;
	}
	else {
		value = 7.787*q + 0.137931;
		return value;
	}
}

LAB RGB2LAB (RGB rgb)
{
	double RGB[3];
	double XYZ[3];
	LAB Lab;
	double adapt[3];
//	double trans[3];
//	double transf[3];
//	double newXYZ[3];
//	double newRGB[3];
	double value;
	//maybe change to global, XYZ[0] = X_value

	adapt[0] = 0.950467;
	adapt[1] = 1.000000;
	adapt[2] = 1.088969;

	RGB[0] = rgb.r * 0.003922;
	RGB[1] = rgb.g * 0.003922;
	RGB[2] = rgb.b * 0.003922;

	XYZ[0] = 0.412424 * RGB[0] + 0.357579 * RGB[1] + 0.180464 * RGB[2];
	XYZ[1] = 0.212656 * RGB[0] + 0.715158 * RGB[1] + 0.0721856 * RGB[2];
	XYZ[2] = 0.0193324 * RGB[0] + 0.119193 * RGB[1] + 0.950444 * RGB[2];

	Lab.l = 116 * H( XYZ[1] / adapt[1] ) - 16;
	Lab.a = 500 * ( H( XYZ[0] / adapt[0] ) - H ( XYZ[1] / adapt[1] ) );
	Lab.b = 200 * ( H( XYZ[1] / adapt[1] ) - H ( XYZ[2] / adapt[2] ) );

	return Lab;
}
