using MathNet.Numerics.LinearAlgebra;

class OptimizationMethods
{
    const double eps = 0.001;

    public double f1(double x1, double x2)
    {
        return 100 * Math.Pow(x2 - x1, 2) + Math.Pow(1 - x1, 2);
    }

    public double f2(double x, double y)
    {
        return Math.Exp(-Math.Pow(x - 3, 2) - Math.Pow(y - 1, 2) / 3) +
                       2 * Math.Exp(-Math.Pow(x - 2, 2) / 2 - Math.Pow(y - 2, 2));
    }

    public (double, double, double[], double[]) gaussMethod(Func<double, double, double> f, Func<double, double, double[]> f_deriv, double x0, double y0)
    {
        double x = 0;
        double y = 0;
        double[] grad = f_deriv(x0, y0);
        double[] stepsx = new double[1000];
        double[] stepsy = new double[1000];
        int step = 0;
        do
        {
            y0 = y;
            x0 = x;
            grad = f_deriv(x, y);
            x -= eps / 10 * (-grad[0]);
            grad = f_deriv(x, y);
            y -= eps / 10 * (-grad[1]);
            stepsx.SetValue(x, step);
            stepsx.SetValue(y, step);
        } while (Math.Abs(x - x0) > eps || Math.Abs(y - y0) > eps);

        return (x, y, stepsx, stepsy);
    }

    double[] f1_deriv(double x, double y)
    {
        double[] delta_f1 = [0, 0];
        delta_f1[0] = 202 * x - 200 * y - 2;
        delta_f1[1] = -200 * x + 200 * y;
        return delta_f1;
    }
    double[] f2_deriv(double x, double y)
    {
        double[] delta_f2 = [0, 0];
        delta_f2[0] = Math.Exp(-Math.Pow(x - 3, 2) / 1 - Math.Pow(y - 1, 2) / 3) * (-2 * (x - 3) / 1)
                     + 2 * Math.Exp(-Math.Pow(x - 2, 2) / 2 - Math.Pow(y - 2, 2) / 1) * (-2 * (x - 2) / 2);

        delta_f2[1] = Math.Exp(-Math.Pow(x - 3, 2) / 1 - Math.Pow(y - 1, 2) / 3) * (-2 * (y - 1) / 3)
                     + 2 * Math.Exp(-Math.Pow(x - 2, 2) / 2 - Math.Pow(y - 2, 2) / 1) * (-2 * (y - 2) / 1);
        return delta_f2;
    }


    double f2_boost_deriv(double l, double[] coords, Matrix<double> s)
    {

        double u1 = coords[0] + l * s[0, 0] - 3;
        double u2 = coords[1] + l * s[1, 0] - 1;
        double u3 = coords[0] + l * s[0, 0] - 2;
        double u4 = coords[1] + l * s[1, 0] - 2;

        double term1 = -2 * s[0, 0] * u1 * Math.Exp(-Math.Pow(u1, 2) - Math.Pow(u2, 2) / 3);
        double term2 = -2 * s[1, 0] * u2 / 3 * Math.Exp(-Math.Pow(u1, 2) - Math.Pow(u2, 2) / 3);
        double term3 = -2 * s[0, 0] * u3 * Math.Exp(-Math.Pow(u3, 2) / 2 - Math.Pow(u4, 2));
        double term4 = -4 * s[1, 0] * u4 * Math.Exp(-Math.Pow(u3, 2) / 2 - Math.Pow(u4, 2));

        return term1 + term2 + term3 + term4;
    }

    double minimalFinder(Func<double, double[], Matrix<double>, double> f_deriv, double[] coords, Matrix<double> s, double lambda)
    {
        double l = lambda;
        double grad = 1;

        for (int i = 0; i < 1000; i++)
        {
            grad = f_deriv(l, coords, s);
            if (Math.Abs(grad) < eps)
            {
                return l;
            }
            l -= eps / 10 * grad;

            if (i == 999)
            {
                return lambda;
            }
        }
        return l;
    }

    public (double, double, double[], double[]) ConjugateGradients(int type, double x, double y)
    {
        Matrix<double> grad0 = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
        Matrix<double> grad = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
        Matrix<double> s0 = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
        Matrix<double> s = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
        double lambda = eps;
        double omega = 0;
        double[] stepsx = new double[1000];
        double[] stepsy = new double[1000];
        int step = 0;

        if (type == 0)
        {
            Matrix<double> gesse = Matrix<double>.Build.DenseOfColumnArrays([[101, -100], [-100, 100]]);
            grad0.SetColumn(0, f1_deriv(x, y));
            s0 = -grad0;
            do
            {
                lambda = -((grad0.Transpose() * s0)[0, 0] / (s0.Transpose() * 2 * gesse * s0)[0, 0]);
                x += lambda * s0[0, 0];
                y += lambda * s0[1, 0];
                stepsx.SetValue(x, step);
                stepsy.SetValue(y, step);
                grad.SetColumn(0, f1_deriv(x, y));
                omega = (grad.Transpose() * gesse * s0)[0, 0] / (s0.Transpose() * gesse * s0)[0, 0];
                s = -grad + omega * s0;
                s0 = s;
                grad0 = grad;
                step += 1;
            } while (Math.Abs(grad0[0, 0]) >= eps || Math.Abs(grad0[1, 0]) >= eps);
        }
        else
        {
            grad0.SetColumn(0, f2_deriv(x, y));
            s0 = -grad0;
            double gg = 0;
            do
            {
                gg = grad0.TransposeAndMultiply(grad0)[0, 0];
                lambda = minimalFinder(f2_boost_deriv, [x, y], s, lambda);
                x += lambda * s0[0, 0];
                y += lambda * s0[1, 0];
                stepsx.SetValue(x, step);
                stepsy.SetValue(y, step);
                grad.SetColumn(0, f2_deriv(x, y));
                omega = (grad.TransposeThisAndMultiply(grad))[0, 0] / gg;
                s = -grad + omega * s0;
                s0 = s;
                grad0 = grad;
                step += 1;
            } while (Math.Sqrt(grad.TransposeAndMultiply(grad)[0, 0]) > eps);
        }
        return (x, y, stepsx, stepsy);
    }

}