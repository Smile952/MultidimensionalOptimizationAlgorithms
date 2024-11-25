

using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Text.Json;

namespace MultidimensionalSearch_Visual
{
    public partial class Form1 : Form
    {
        double x = 10;
        double y = 0;
        OptimizationMethods methods;
        List<double> xVal = new List<double>();
        List<double> yVal = new List<double>();
        List<Krug> krugs = new List<Krug>();


        public Form1()
        {
            InitializeComponent();
        }

        public void draw(object sender, PaintEventArgs e)
        {
            int val = 10;
            methods = new OptimizationMethods(x, y, 0);
            (xVal, yVal) = methods.getGausseAnswer();

            double xAns = xVal.Last<double>();
            double yAns = yVal.Last<double>();
            krugs = new List<Krug>();
            double xStart = xVal.Last<double>();
            double yStart = yVal.Last<double>();
            Pen pen = new Pen(Color.Red, 1);
            for (int i = xVal.Count - 1; i >= 0; i--)
            {
                float x = (float)(200 - Math.Abs(xVal[0]));
                float y = (float)(200 - Math.Abs(yVal[0]));
                float d1 = 5 * (float)Math.Abs(methods.f1(xVal[i], yVal[i]))*val;
                float d2 = 5 * (float)Math.Abs(methods.f1(xVal[i], yVal[i]))*val;
                e.Graphics.DrawEllipse(pen, x, y, d1, d2);  
            }
        }
    }
}
public class Krug
{
    public double d1 { set; get; }
    public double d2 { set; get; }
    public double x { set; get; }
    public double y { set; get; }
    public Krug(double d1, double d2, double x, double y)
    {
        this.d1 = d1;
        this.d2 = d2;
        this.x = x;
        this.y = y;
    }
}


