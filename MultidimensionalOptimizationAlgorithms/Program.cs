using MathNet.Numerics.LinearAlgebra;

double eps = 0.001;

double f1(double x1, double x2)
{
    return 100 * Math.Pow(x2 - x1, 2) + Math.Pow(1 - x1, 2); 
}

double f2(double x, double y)
{
    return Math.Exp(-Math.Pow(x - 3, 2) / 1 - Math.Pow(y - 1, 2) / 3) +
                   2 * Math.Exp(-Math.Pow(x - 2, 2) / 2 - Math.Pow(y - 2, 2) / 1);
}
double[] gaussMethod(Func<double, double, double> f, Func<double, double, double[]> f_deriv, double x0, double y0)
{
    double x = 0;
    double y = 0;
    double[] grad = f_deriv(x0, y0);
    do
    {
        y0 = y;
        x0 = x;
        grad = f_deriv(x, y);
        x -= eps/10 * (-grad[0]);
        grad = f_deriv(x, y);
        y -= eps/10 * (-grad[1]);
    } while (Math.Abs(x - x0) > eps || Math.Abs(y - y0) > eps);

    return [x, y];
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
    delta_f2[0] = Math.Exp(-Math.Pow(x - 3, 2) / 1 - Math.Pow(y - 1, 2) / 3) * (-2 * (x - 3) / 1)
                 + 2 * Math.Exp(-Math.Pow(x - 2, 2) / 2 - Math.Pow(y - 2, 2) / 1) * (-2 * (x - 2) / 2);

    delta_f2[1] = Math.Exp(-Math.Pow(x - 3, 2) / 1 - Math.Pow(y - 1, 2) / 3) * (-2 * (y - 1) / 3)
                 + 2 * Math.Exp(-Math.Pow(x - 2, 2) / 2 - Math.Pow(y - 2, 2) / 1) * (-2 * (y - 2) / 1);
    return delta_f2;
}


double[] ConjugateGradients(Func<double, double, double[]> f_deriv, double x, double y)
{
    Matrix<double> grad0 = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
    Matrix<double> grad = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
    Matrix<double> s0 = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
    Matrix<double> s = Matrix<double>.Build.DenseOfColumnArrays([1, 1]);
    double lambda = eps;
    double omega = 0;
    grad0.SetColumn(0, f_deriv(x, y));
    s0 = -grad0;
    double gg = 0;
    do
    {
        gg = grad0.TransposeAndMultiply(grad0)[0, 0];
        lambda = gg / s0.TransposeAndMultiply(grad0)[0, 0];
        x +=  s0[0, 0];
        y +=  s0[1, 0];
        grad.SetColumn(0, f_deriv(x, y));
        omega = (grad.TransposeThisAndMultiply(grad))[0, 0] / gg;
        s = -grad + omega * s0;
        s0 = s;
        grad0 = grad;
    } while (Math.Sqrt(grad.TransposeAndMultiply(grad)[0, 0]) > eps);
    return [x, y];
}

double[] ans = ConjugateGradients(f2_deriv, 0, 0);

Console.WriteLine(ans[0].ToString() + " " + ans[1].ToString());
Console.WriteLine(f2(ans[0], ans[1]));

ans = gaussMethod(f2, f2_deriv, 0, 0);
Console.WriteLine(ans[0].ToString() + " " + ans[1].ToString());
Console.WriteLine(f2(ans[0], ans[1]));