using MathNet.Numerics.LinearAlgebra;
using System.Diagnostics;
using System.Windows.Forms.VisualStyles;

class OptimizationMethods
{
    const double eps = 0.001;

    private double x;
    private double y;

    public int type = 0;

    public OptimizationMethods(double x, double y, int type)
    {
        this.x = x;
        this.y = y;
        this.type = type;
    }

    public void setType(int type)
    {
        this.type = type;   
    }

    public void setCoords(double x, double y)
    {
        this.x = x;
        this.y = y;
    }

    public double f1(double x1, double x2)
    {
        return 100 * Math.Pow(x2 - x1, 2) + Math.Pow(1 - x1, 2);
    }

    public double f2(double x, double y)
    {
        return Math.Exp(-Math.Pow(x - 3, 2) - Math.Pow(y - 1, 2) / 3) +
                       2 * Math.Exp(-Math.Pow(x - 2, 2) / 2 - Math.Pow(y - 2, 2));
    }
    public (List<double>, List<double>) gaussMethod(Func<double, double, double> f, Func<double, double, double[]> f_deriv, double x0, double y0)
    {
        double gausX = x0;
        double gausY = y0;
        double gausX_0 = x0;    
        double gausY_0 = y0;    
        double[] grad = f_deriv(x0, y0);
        List<double> stepsX = new List<double>();
        List<double> stepsY = new List<double>();
        stepsX.Add(gausX_0);
        stepsY.Add(gausY_0);
        do
        {
            gausY_0 = gausY;
            gausX_0 = gausX;
            grad = f_deriv(gausY, gausY);
            gausX -= eps  * grad[0];
            grad = f_deriv(gausX, gausY);
            gausY -= eps  * grad[1];
            stepsX.Add(gausX);
            stepsY.Add(gausY);
        } while (Math.Abs(f(gausX, gausY)) > eps);

        return (stepsX, stepsY);
    }

    public double[] f1_deriv(double x, double y)
    {
        double[] delta_f1 = [0, 0];
        delta_f1[0] = 202 * x - 200 * y - 2;
        delta_f1[1] = -200 * x + 200 * y;
        return delta_f1;
    }
    public double[] f2_deriv(double x, double y)
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

    public (List<double>, List<double>) ConjugateGradients(double x, double y)
    {
        Matrix<double> grad0 = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
        Matrix<double> grad = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
        Matrix<double> s0 = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
        Matrix<double> s = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
        double lambda = eps;
        double omega = 0;
        List<double> stepsX = new List<double>();
        List<double> stepsY = new List<double>();
        double conX = x;
        double conY = y;    
        stepsX.Add(conX);
        stepsY.Add(conY);
        if (type == 0)
        {
            Matrix<double> gesse = Matrix<double>.Build.DenseOfColumnArrays([[101, -100], [-100, 100]]);
            grad0.SetColumn(0, f1_deriv(conX, conY));
            s0 = -grad0;
            do
            {
                lambda = -((grad0.Transpose() * s0)[0, 0] / (s0.Transpose() * 2 * gesse * s0)[0, 0]);
                conX += lambda * s0[0, 0];
                conY += lambda * s0[1, 0];
                stepsX.Add(conX);
                stepsY.Add(conY);
                grad.SetColumn(0, f1_deriv(conX, conY));
                omega = (grad.Transpose() * gesse * s0)[0, 0] / (s0.Transpose() * gesse * s0)[0, 0];
                s = -grad + omega * s0;
                s0 = s;
                grad0 = grad;
            } while (Math.Abs(f1(conX, conY)) >= eps);
        }
        else
        {
            grad0.SetColumn(0, f2_deriv(conX, conY));
            s0 = -grad0;
            double gg = 0;
            do
            {
                gg = grad0.TransposeAndMultiply(grad0)[0, 0];
                lambda = minimalFinder(f2_boost_deriv, [conX, conY], s, lambda);
                conX+= lambda * s0[0, 0];
                conY += lambda * s0[1, 0];
                stepsX.Add(conX);
                stepsY.Add(conY);
                grad.SetColumn(0, f2_deriv(conX, conY));
                omega = grad.TransposeThisAndMultiply(grad)[0, 0] / gg;
                s = -grad + omega * s0;
                s0 = s;
                grad0 = grad;
            } while (Math.Abs(f2(conX, conY)) > eps);
        }
        return (stepsX, stepsY);
    }

    public (List<double>, List<double>) getConGradAnswer()
    {
        return ConjugateGradients(x, y);
    }

    public (List<double>, List<double>) getGausseAnswer()
    {
        if (type == 0)
        {
            return gaussMethod(f1, f1_deriv, x, y);
        }
        else
            return gaussMethod(f2, f2_deriv, x, y);
    }
}