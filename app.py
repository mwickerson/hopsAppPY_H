"""Hops flask middleware example"""
from flask import Flask
import ghhops_server as hs
import rhino3dm


# register hops app as middleware
app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)


# flask app can be used for other stuff drectly
@app.route("/help")
def help():
    return "Welcome to Grashopper Hops for CPython!"

"""
@hops.component(
    "/binmult",
    inputs=[hs.HopsNumber("A"), hs.HopsNumber("B")],
    outputs=[hs.HopsNumber("Multiply")],
)
def BinaryMultiply(a: float, b: float):
    return a * b


@hops.component(
    "/add",
    name="Add",
    nickname="Add",
    description="Add numbers with CPython",
    inputs=[
        hs.HopsNumber("A", "A", "First number"),
        hs.HopsNumber("B", "B", "Second number"),
    ],
    outputs=[hs.HopsNumber("Sum", "S", "A + B")]
)
def add(a: float, b: float):
    return a + b


@hops.component(
    "/pointat",
    name="PointAt",
    nickname="PtAt",
    description="Get point along curve",
    icon="pointat.png",
    inputs=[
        hs.HopsCurve("Curve", "C", "Curve to evaluate"),
        hs.HopsNumber("t", "t", "Parameter on Curve to evaluate")
    ],
    outputs=[hs.HopsPoint("P", "P", "Point on curve at t")]
)
def pointat(curve: rhino3dm.Curve, t=0.0):
    return curve.PointAt(t)


@hops.component(
    "/srf4pt",
    name="4Point Surface",
    nickname="Srf4Pt",
    description="Create ruled surface from four points",
    inputs=[
        hs.HopsPoint("Corner A", "A", "First corner"),
        hs.HopsPoint("Corner B", "B", "Second corner"),
        hs.HopsPoint("Corner C", "C", "Third corner"),
        hs.HopsPoint("Corner D", "D", "Fourth corner")
    ],
    outputs=[hs.HopsSurface("Surface", "S", "Resulting surface")]
)
def ruled_surface(a: rhino3dm.Point3d,
                  b: rhino3dm.Point3d,
                  c: rhino3dm.Point3d,
                  d: rhino3dm.Point3d):
    edge1 = rhino3dm.LineCurve(a, b)
    edge2 = rhino3dm.LineCurve(c, d)
    return rhino3dm.NurbsSurface.CreateRuledSurface(edge1, edge2)
"""
# sphere sin
@hops.component(
    "/sphere_sin3",
    name="Sphere Sin",
    nickname="SphSin",
    description="Create sphere with sin function",
    inputs=[
        hs.HopsInteger("num", "n", "Number of spheres"),
        hs.HopsNumber("spacing", "s", "Spacing between spheres"),
        hs.HopsNumber("radius", "r", "Radius of spheres"),
    ],
    outputs=[hs.HopsBrep("Spheres", "S", "Resulting spheres")]

)
def sphere_sin3(num: int, spacing: float, radius: float):
    import math
    items = []
    for i in range(0,num):
        for j in range(0,num):
            pt = rhino3dm.Point3d(i * spacing, j * spacing, 0)
            rad = radius + math.sin(i * j) * radius
            sphere = rhino3dm.Sphere(pt, rad)
            items.append(sphere.ToBrep())
    return items



        
    

    





"""
███╗   ███╗ █████╗ ████████╗██████╗ ██╗      ██████╗ ████████╗██╗     ██╗██████╗ 
████╗ ████║██╔══██╗╚══██╔══╝██╔══██╗██║     ██╔═══██╗╚══██╔══╝██║     ██║██╔══██╗
██╔████╔██║███████║   ██║   ██████╔╝██║     ██║   ██║   ██║   ██║     ██║██████╔╝
██║╚██╔╝██║██╔══██║   ██║   ██╔═══╝ ██║     ██║   ██║   ██║   ██║     ██║██╔══██╗
██║ ╚═╝ ██║██║  ██║   ██║   ██║     ███████╗╚██████╔╝   ██║   ███████╗██║██████╔╝
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝ ╚═════╝    ╚═╝   ╚══════╝╚═╝╚═════╝ 
"""

# matplotlib    and numpy for plotting

# linear regrassion using least squares nethod
# w1x + w2 = y
# w1 = (x1 * y - x2 * y) / (x1^2 + x2^2)
# x co-ordinates
@hops.component(
    "/linear_regression",
    name=("Linear Regression"),
    description=("Linear Regression"),
    category="numpy",
    subcategory="array",
    inputs=[
        hs.HopsNumber("xList", "xList", "xList", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("yList", "yList", "yList", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsNumber("linear_regression", "linear_regression", "linear_regression", access = hs.HopsParamAccess.LIST),
    ]
)
def linear_regression(xList, yList):
    """
    Linear Regression
    """
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.array(xList)
    y = np.array(yList)
    x1 = x[0]
    x2 = x[1]
    y1 = y[0]
    y2 = y[1]
    w1 = (x1 * y1 - x2 * y2) / (x1**2 + x2**2)
    w2 = (x1 * y2 - x2 * y1) / (x1**2 + x2**2)
    linear_regression = [w1, w2]
    return linear_regression    

@hops.component(
    "/linear_regression_plot",
    name=("Linear Regression Plot"),
    description=("Linear Regression Plot"),
    category="numpy",
    subcategory="array",
    inputs=[
        hs.HopsNumber("xList", "xList", "xList", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("yList", "yList", "yList", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsBoolean("linear_regression_plot", "linear_regression_plot", "linear_regression_plot", access = hs.HopsParamAccess.LIST),
    ]
)
def linear_regression_plot(xList, yList):
    
    #Linear Regression Plot
    
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.array(xList)
    A = np.array([x, np.ones(len(x))])
    # linearly generated sequence
    y = np.array(yList)
    # obtaining the parameters of the regression line
    w = np.linalg.lstsq(A.T, y)[0]
    # plotting the regression line
    line = w[0] * x + w[1] # regression line
    plt.plot(x, line, 'r-', label='Linear regression')
    plt.plot(x, y, 'o', label='Original data')
    plt.show()
    return True

# matplotlib functions for plotting
@hops.component(
    "/plot_scatter1",
    name=("Plot Scatter"),
    description=("Plot Scatter"),
    category="numpy",
    subcategory="array",
    inputs=[
        hs.HopsNumber("xList", "xList", "xList", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("yList", "yList", "yList", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsBoolean("plot_scatter", "plot_scatter", "plot_scatter", access = hs.HopsParamAccess.LIST),
    ]
)
def plot_scatter(xList, yList):

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.array(xList)
    y = np.array(yList)
    plt.scatter(x, y)
    plt.show()
    return True

# matplotlib functions for plotting
@hops.component(
    "/plot_graph",
    name=("Plot Graph"),
    description=("Plot Graph"),
    category="numpy",
    subcategory="array",
    inputs=[
        hs.HopsNumber("xList", "xList", "xList", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("yList", "yList", "yList", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsBoolean("plot_graph", "plot_graph", "plot_graph", access = hs.HopsParamAccess.LIST),
    ]
)
def plot_graph(xList, yList):
    
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.array(xList)
        y = np.array(yList)
        plt.plot(x, y)
        plt.show()
        return True

# bar graph for plotting matplotlib
@hops.component(
    "/plot_bar",
    name=("Plot Bar"),
    description=("Plot Bar"),
    category="numpy",
    subcategory="array",
    inputs=[
        hs.HopsNumber("xList", "xList", "xList", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("yList", "yList", "yList", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsBoolean("plot_bar", "plot_bar", "plot_bar", access = hs.HopsParamAccess.LIST),
    ]
)
def plot_bar(xList, yList):
        
            import matplotlib.pyplot as plt
            import numpy as np
            x = np.array(xList)
            y = np.array(yList)
            plt.bar(x, y)
            plt.show()
            return True

@hops.component(
    "/plot_pie",
    name=("Plot Pie"),
    description=("Plot Pie"),
    category="numpy",
    subcategory="array",
    inputs=[
        hs.HopsNumber("xList", "xList", "xList", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("yList", "yList", "yList", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsBoolean("plot_pie", "plot_pie", "plot_pie", access = hs.HopsParamAccess.LIST),
    ]
)
def plot_pie(xList, yList):
            import matplotlib.pyplot as plt
            import numpy as np
            x = np.array(xList)
            y = np.array(yList)
            plt.pie(y, labels=x)
            plt.show()
            return True

@hops.component(
    "/plot_donut",
    name=("Plot Donut"),
    description=("Plot Donut"),
    category="numpy",
    subcategory="array",
    inputs=[
        hs.HopsNumber("xList", "xList", "xList", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("yList", "yList", "yList", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsBoolean("plot_donut", "plot_donut", "plot_donut", access = hs.HopsParamAccess.LIST),
    ]
)
def plot_donut(xList, yList):
            import matplotlib.pyplot as plt
            import numpy as np
            x = np.array(xList)
            y = np.array(yList)
            plt.pie(y, labels=x, autopct='%1.1f%%', startangle=90)
            plt.show()
            return True

@hops.component(
    "/plot_scatter_3d",
    name=("Plot Scatter 3D"),
    description=("Plot Scatter 3D"),
    category="numpy",
    subcategory="array",
    inputs=[
        hs.HopsNumber("xList", "xList", "xList", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("yList", "yList", "yList", access = hs.HopsParamAccess.LIST),
        hs.HopsNumber("zList", "zList", "zList", access = hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsBoolean("plot_scatter_3d", "plot_scatter_3d", "plot_scatter_3d", access = hs.HopsParamAccess.LIST),
    ]
)
def plot_scatter_3d(xList, yList, zList):
            import matplotlib.pyplot as plt
            import numpy as np
            x = np.array(xList)
            y = np.array(yList)
            z = np.array(zList)
            plt.scatter(x, y, z)
            plt.show()
            return True




if __name__ == "__main__":
    app.run(debug=True)
