﻿using System;
using MathNet.Numerics.LinearAlgebra;

double eps = 0.01;

int num_iter = 0;

double f1(double x1, double x2)
{
    return 100 * Math.Pow(x2 - x1, 2) + Math.Pow(1 - x1, 2); 
}

double f2(double x, double y)
{
    return Math.Exp(-Math.Pow(x - 3, 2) - Math.Pow((y - 1) / 3, 2)) + 2*Math.Exp(-Math.Pow((x - 2) / 2, 2) - Math.Pow(y - 2, 2));
}
double[] gaussMethod(Delegate f, double x0, double y0)
{
    double dx = eps / 2;
    double dy = dx;

    double z0 = 0;
    double z0_global = 0;
    double z = (double)f.DynamicInvoke([x0, y0]);
    do
    {
        z0_global = f1(x0, y0);
        while ((z - z0) > 0)
        {
            double z1_x = (double)f.DynamicInvoke(x0 + dx, y0);
            double z2_x = (double)f.DynamicInvoke(x0 - dx, y0);
            
            if(z > z2_x && z > z1_x)
            {
                if(z1_x > z2_x)
                {
                    x0 += eps;
                    num_iter++;
                }
                else
                {
                    x0 -= eps;
                    num_iter++;
                }
            }
            else if (z > z1_x)
            {
                x0 += eps;
                num_iter++;
            }
            else if (z > z2_x)
            {
                x0 -= eps;
                num_iter++;
            }
            z0 = z;
        }
        z0 = 0;
        z = (double)f.DynamicInvoke(x0, y0);
        while ((z - z0) > 0)
        {
            double z1_y = f1(x0, y0 + dy);
            double z2_y = f1(x0, y0 - dy);

            if (z > z2_y && z > z1_y)
            {
                if (z1_y > z2_y)
                {
                    y0 += eps;
                    num_iter++;
                }
                else
                {
                    y0 -= eps;
                    num_iter++; 
                }
            }
            else if (z > z1_y)
            {
                y0 += eps;
                num_iter++;
            }
            else if (z > z2_y)
            {
                y0 -= eps;
                num_iter++;
            }
            
            z0 = z;
        }
        z = (double)f.DynamicInvoke(x0, y0);
    } while ((z - z0_global) > eps);

    return [x0, y0, z];
}

double[] f1_deriv(double x,  double y)
{
    double[] delta_f1 = [0, 0];
    delta_f1[0] = 202 * x - 200 * y - 2;
    delta_f1[1] = -200 * x + 200 * y;
    return delta_f1;
}
double[] f2_deriv(double x, double y)
{
    double[] delta_f2 = [0, 0];
    delta_f2[0] = -2 * (x - 3) * Math.Exp(-(Math.Pow(x - 3, 2) - Math.Pow((y - 1) / 3, 2))) - (x-2)*Math.Exp(-(Math.Pow((x-2)/2, 2) - Math.Pow(y-2, 2)));
    delta_f2[1] = -(2 * (y-1)/9) * Math.Exp(-(Math.Pow(x - 3, 2) - Math.Pow((y - 1) / 3, 2))) - 4*(y-2)*Math.Exp(-(Math.Pow((x-2)/2, 2) - Math.Pow(y-2, 2)));
    return delta_f2;
}

double[] ConjugateGradients(double x, double y)
{
    Matrix<double> grad0 = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
    Matrix<double> grad = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
    Matrix<double> s0 = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
    Matrix<double> s = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
    Matrix<double> gesse = Matrix<double>.Build.DenseOfColumnArrays([[101, -100], [-100, 100]]);
    double lambda = 0;
    double omega = 0;
    grad0.SetColumn(0, f1_deriv(x, y));
    s0 = -grad0;
    do
    {
        lambda = -((grad0.Transpose() * s0)[0, 0] / (s0.Transpose() * (2*gesse) * s0)[0, 0]);
        x += lambda * s0[0, 0];
        y += lambda * s0[1, 0];

        grad.SetColumn(0, f1_deriv(x, y));
        omega = (grad.Transpose() * gesse * s0)[0, 0] / (s0.Transpose() * gesse * s0)[0, 0];
        s = -grad + omega * s0;
        Console.WriteLine(s);
        s0 = s;
        grad0 = grad;
    } while (Math.Abs(s[0, 0]) > eps || Math.Abs(s[1, 0]) > eps);
    return [x, y];
}