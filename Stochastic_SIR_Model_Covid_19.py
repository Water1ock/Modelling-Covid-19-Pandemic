import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    return math, mo, np, plt


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Stochastic model of COVID infection excluding recovery

        We're going to build an initial simulation. We will keep adding to it until we get to the full SIR model.

        The goals are to
        - get better with the methodology of model building
        - reinforce understanding of differential equations
        - get exposure to different coding patterns

        ## The situation
        -  $N$ students in total
        -  $S_k$ is the number of uninfected (i.e. **S**usceptible) students on day $k$
        - Probability of an uninfected person getting infected on a given day is $\lambda$


        In this model, each day, a certain number of people. The output of the model is the timecourse of infected people over time.

        We'll start with a highly simplified model where a proportion $p = 1 - \lambda$  (e.g. $35 \%$) of the remaining people get infected every day. Therefore, the infection rate is **independent** of the proportion of infected people.

        !!! warning "Question"
            Why is this independence a bad assumption?

        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.accordion(
        {"Answer" : 
    mo.md(r""" 
    The number of infected people should depend on both the number of suspectible people as well as the number of infected people.

    To see this, consider a small classroom of $20$ people and the following two cases:
    1. on day $k$, $15$ infected people went to class;
    2. on day $k$, $1$ infected person went to class;

    If you were in the class, would it be more likely to get infected in case $1$ or case $2$ on day $k+1$?
    """)
                 })
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Step 1. Build the hyperparameters

        - Hyperparameters are quantities that change the problem itself. Most programming / AI workflows will have some hyperparameters.
        - It's often useful to put them all in a single dictionary, which can be passed to the functions that run the simulation
        - That way, if you want to test different groups of hyperparameters, you can pack them all in different dictionaries and they won't interfere with each other.

        We've built the hyperparameter dictionary below. The actual values for the different hyperparameters are in sliders located above the first plot. So you can change them
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    # these sliders are visualised and modifiable a few slides down!!!
    n_students = mo.ui.slider(5,100,1, show_value=True, value=20, label = "Number of students")
    infection_rate = mo.ui.slider(0,1,0.01, show_value=True, value=0.1,  label = "infection rate")
    simulation_length = mo.ui.slider(1,100,1, show_value=True, value=30,  label = "simulation length (days)")
    repeats = mo.ui.slider(1,10,1, show_value=True, value=3,  label = "repeats")  

    return infection_rate, n_students, repeats, simulation_length


@app.cell
def __(infection_rate, n_students, repeats, simulation_length):
    hyperparameters = {"N": n_students.value, "λ": infection_rate.value, "days": simulation_length.value, "repeats": repeats.value}
    return (hyperparameters,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Modelling infections as random variables

        - Does a given person get infected on a given day? This is a quantitative question about an experiment: i.e. a random variable. It takes the form of a true/false question. Hence we can model it probabilistically with a **Bernoulli distribution**

        - How many susceptible people get infected on a given day? This is the sum of $S$ true/false questions, each with the same probability. IE $S$ identical Bernoulli distributions. Hence, it is a **Binomial distribution**.

        !!! warning "Question"
            What assumptions did I make in the text box above?
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.accordion(
        {"Answer" : 
    mo.md(r""" - We assumed that the question: "did you get infected on a given day?" is **independent** of the answers of all the other people. A bit like assuming that the result of a coin toss is independent of the previous coin toss. 
    	- However in this case, there may be some dependence between students in the class. For instance, consider three students who are friends with each other. If on day $k$ one of the three friends gets infected, the probability of the two other students getting infected the day after is probably higher that that of another person who is not in the friends group.
    """)
                 })
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""- Now we are going to simulate infections over time. Inspect the `run_simulation` code below, carefully. Make sure you understand what is going on""")
    return


@app.cell
def __(np):
    def run_simulation(hyperparams):

        def B(susceptible):
            return np.random.binomial(susceptible, 1 - hyperparams["λ"])  # correct binomial function
        
        S = np.zeros((hyperparams["days"], hyperparams["repeats"]))  # rows = days, cols = repeats, entries = number of susceptible people
        
        S[0, :] = hyperparams["N"]
        for repeat in range(S.shape[1]):
            for i in range(1, S.shape[0]):
                S[i, repeat] = B(S[i-1, repeat])  # random number distributed according to B(previous day susceptible people, repeat)
        
        return S
    return (run_simulation,)


@app.cell(hide_code=True)
def __(mo):
    mo.accordion(
        {"Hint" : 
    mo.md(r""" - `B(susceptible)` defines the binomial probability distribution using the number of Susceptible people and the probability of infection as parameters;
    - `S` is defined to store tha values of Susceptible people for each day (each row a diff day) ad each trial (each column a different trial);
    - `S[0,:]` the first value of $S$ for day $1$ is equal to $N$, i.e., all students are susceptible, there is no infected person yet on day $1$;
    - then we loop for each day, and find the number of infected people everyday depending on the number of susceptible people $S$ from the day before;
    - `repeats`, is the number of simulations we are running at once.

    The result of the simulations is $S$ output below.
    """)
                 })
    return


@app.cell
def __(hyperparameters, run_simulation):
    S = run_simulation(hyperparameters)
    S
    return (S,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        - Each column stores the number of susceptible people each day for a differet trial (i.e., for `repeat = 2` we have $2$ simulations, hence $2$ columns).
        - Now we can plot the simulation (see code below):
        """
    )
    return


@app.cell
def __(infection_rate, n_students, repeats, simulation_length):
    (n_students ,
    infection_rate, 
    simulation_length, 
    repeats) #these sliders were defined a few cells above!
    return


@app.cell(hide_code=True)
def __(hyperparameters, mo, np, plt, run_simulation):
    def plot_simulation(h):
        S = run_simulation(h)
        I = h["N"] - S

        labels = [f"repeat {i+1}" for i in range(h["repeats"])]  # String interpolation equivalent in Python
        labels = np.array(labels).reshape(1, -1)  # labels need to be a 1 x n matrix for some silly reason
        
        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        
        # Susceptible plot
        for repeat in range(h["repeats"]):
            ax[0].scatter(np.arange(1, h["days"] + 1), S[:, repeat], label=labels[0, repeat])
        ax[0].set_ylabel("Number of people")
        ax[0].set_xlabel("day")
        ax[0].set_title("Susceptible")
        ax[0].legend()

        # Infected plot
        for repeat in range(h["repeats"]):
            ax[1].scatter(np.arange(1, h["days"] + 1), I[:, repeat], label=None)
        ax[1].set_ylabel("Number of people")
        ax[1].set_xlabel("day")
        ax[1].set_title("Infected")
        
        plt.tight_layout()
        
        return fig
    mo.as_html(plot_simulation(hyperparameters));
    return (plot_simulation,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        !!! warning "Question"
            Explain how each of the parameters affect the inter-trial stochasticity, and why.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.accordion(
        {"Answer" : 
    mo.md(r""" First of all, what is inter-trial stochasticity?
    Inter-trial stochasticity is a another way of saying difference due to random effects between different simulations. If there was no inter-trial stochasticity, then each simulation will give exactly the same output at each step.

    To answer the question, let's take a look at the parameters we have used for the simulation and determine which ones may affect the inter-trial stochasticity.

    The `hyperparameters` are:
    - `repeat` i.e., the number of trials. The more trials we have, the more simulations we run. This doesn't change the statistics of individual repeats. But it does allows us to better estimate the inter-trial stochasticity.

    - Each day `1-λ` is the probability of infection for an individual. Consider the case `λ = 0`. In this case there is no stochasticity in the system, since everybody is immediately infected on day 2. Similarly, if `λ = 1`, nobody gets infected: there is no stochasticity. In fact, the most stochasticity will occur when $\lambda = 0.5$. The probability of an individual getting infected is a Bernoulli ($\sim Bern(\lambda)$). The variance of a Bernoulli is $\lambda(1 - \lambda)$, which is maximised at $\lambda = 0.5$.

    - `N` is the total number of people. If there are a small number of people $e.g. 5$, the actual proportion of people infected could be very far from $1-\lambda$. EG 5 people could get lucky and remain healthy even if $\lambda=0.5$. If there are instead 1000 people, the actual infection rate is likely to be much closer to the expected infection rate (you'd have to be **really** lucky to get no infections in this case). This is an offshoot of the law of large numbers.
    """)
                 })
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Step 2: Mean field approximations

        We can build the deterministic approximants of the stochastic dynamics of infection, using the methods from the lecture.

        #### Option 1:

        - Assume that the expected value of infections happens each day. This is deterministic, given the hyperparameters.

        *(On the kth day, $S_k \times \lambda$ people get infected in expectation. So $S_{k+1} = S_k - \lambda S_k = S_k(1 - \lambda$)*

        *This is recursive. We get $S_k = (1-\lambda)^kS_0$*
        """
    )
    return


@app.cell
def __(np, plot_simulation):
    def calculate_expected_value(hyps):
        λ = hyps["λ"]
        q = 1 - λ
        days = hyps["days"]
        N = hyps["N"]
        return [N * (q ** k) for k in range(1, days + 1)]

    def plot_expected_SI_value(hyps):
        fig = plot_simulation(hyps)
        
        for s in fig.axes:
            expected_SI = calculate_expected_value(hyps)
            s.scatter(np.arange(1, hyps["days"] + 1), expected_SI, linewidth=4, label="expected infections", marker='H', s=60)
        fig.delaxes(fig.axes[1])
        fig.legend()
        return fig

    return calculate_expected_value, plot_expected_SI_value


@app.cell
def __(hyperparameters, plot_simulation):
    fig = plot_simulation(hyperparameters)
    len(fig.axes)
    return (fig,)


@app.cell
def __(hyperparameters, mo, plot_expected_SI_value):
    mo.as_html(plot_expected_SI_value(hyperparameters))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        #### Option 2

        - Calculate continuous time, mean-field approximation.

        In other words, assume that infections are happening constantly, rather than at midnight each day
        """
    )
    return


@app.cell
def __(hyperparameters, mo, np, plot_simulation, plt):
    def calculate_meanfield_approx(hyps):
        λ = hyps["λ"]
        N = hyps["N"]
        
        def approx(t):
            return N * np.exp(-λ * t)
        
        return approx

    def plot_meanfield_SI(hyps):
        fig = plot_simulation(hyps)
        meanfield_approx = calculate_meanfield_approx(hyps)
        
        s = fig.axes[0]
        fig.delaxes(fig.axes[1])
        s.plot(np.arange(1, hyps["days"] + 1), meanfield_approx(np.arange(1, hyps["days"] + 1)), linewidth=4, label="mean field approximation")
        
        # Display the plot
        s.legend()
        plt.tight_layout()
        return fig
        
        return s
    mo.as_html(plot_meanfield_SI(hyperparameters))
    return calculate_meanfield_approx, plot_meanfield_SI


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Interlude
        ## Forward Euler algorithm for solving ODEs

        ### (Because there is a backward Euler too!)

        - We're deriving the Forward Euler algorithm, just to make sure it's solidified in your head!

        - Approximation error on successive steps of the algorithm could compound, leading to a terrible numerical solution that looks nothing like the true solution. This becomes more likely as $\delta t$ increases. 

        - You can explore at what point the Forward Euler starts to diverge on a simple example. Play around with the code.

        ### Quick re-derivation of Forward Euler

        We start with the general form of a first order ODE:
        $$\frac{dx}{dt}(t) = f(x,t)$$

        **Expressing the derivative as a limit**

        $$\frac{dx}{dt}(t) =  f(x(t), t) = \lim_{\delta t \to 0} \frac{x(t + \delta t) - x(t)}{\delta t}$$

        (Note that we usually use $\dot{x}(t)$ instead of $\frac{dx}{dt}(t)$). 

        **Rearranging**:

        $\lim_{\delta t \to 0} x(t + \delta t) = \lim_{\delta t \to 0} \Big[ x(t) + \delta t * f(x(t), t) \Big]$ 

        $x(t + \delta t) \approx  x(t) + \delta t \times f(x(t), t)$

        - The smaller the timestep $\delta t$, the closer we get to equality.
        - *But how small is small enough???*

        ---
        **Euler's method**
        - Start with $x(0)$ *(initial condition)*
        - Choose a fixed, small $\delta t$ *(how small?)*

        $$x(\delta t) = x(0) + (\delta t)f(x(0),0)$$

        $$x(2\delta t) = x(\delta t) + (\delta t)f(x(\delta t),\delta t)$$

        $$\dots$$

        $$x(N\delta t) = x\big((N-1)\delta t\big) + (\delta t) f\big(x((N-1)\delta t), t\Big)$$
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Let's code it!

        Our goal is to implement the Forward Euler algorithm to solve ODEs of the form:

        $$\frac{dx}{dt}(t) = \dot{x}(t) =  f(x(t),t),$$

        where $x(t)$ is a vector, and $f(x,t)$ is an arbitrary function often called the *vector field* (since it takes in vectors, and spits out vectors of the same size) 

        Let's take an arbitrary vector field (coded below) for testing
        """
    )
    return


@app.cell
def __(np):
    def vf(x, time):
        x_dot = np.array([x[1], -x[0], -x[0] * x[2]**2])
        return x_dot
    result = vf([1, 2, 3], 0)
    result
    return result, vf


@app.cell
def __(np):

    # forward_euler_solve(f::Function, x₀, t₀, tₑ, δt)
    # f should be a function that accepts a vector x₀ of initial conditions, and a function `f` that takes an input state and a time: ie `f(x::Vector,time::Number)`

    # - t₀ is the starting time 
    # - tₑ is the end time
    # - dt is the step size

    def forward_euler_solve(f, x0, t0, te, dt):
        timepoints = np.arange(t0, te, dt)
        xs = np.zeros((len(x0), len(timepoints)))
        xs[:, 0] = x0
        
        def populate(i):
            xs[:, i] = xs[:, i - 1] + dt * f(xs[:, i - 1], timepoints[i - 1])  # the forward euler step
        
        for i in range(1, len(timepoints)):
            populate(i)
        
        return timepoints, xs.T
    return (forward_euler_solve,)


@app.cell
def __(forward_euler_solve, mo, plt, vf):
    timepoints, xs = forward_euler_solve(vf, [1.0, 2, 3], 0, 10, 0.01)
    plt.plot(timepoints, xs)  
    plt.xlabel('Time')
    plt.ylabel('State x(t)')
    mo.as_html(plt.show())
    return timepoints, xs


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### When does approximation error compound and give a 'nonsense' solution?

        **Let's approach this question with an example where we have the ground truth**

        $$\frac{dx}{dt} =-2.5 x(t)$$

        True solution:
        $$x(t) = x(0) \exp(-2.5 t)$$

        Euler step:
        $$x(t + \delta t ) = x(t) - (\delta t) 2.5 x(t)$$
        """
    )
    return


@app.cell
def __():
    def ground_truth_vf(x, t):
        x_dot = -2.5 * x
        return x_dot
    return (ground_truth_vf,)


@app.cell
def __(mo):
    # ` tstop = ` $(@bind tstop Slider(10:100;default=20, show_value=true))

    # `δt = ` $(@bind δt Slider(0.01:0.01:2; default=0.6, show_value=true))

    # ` x0 = ` $(@bind x₀ Slider(0:10; default=8, show_value=true))

    tstop = mo.ui.slider(10,100,1,value=20,show_value=True, label = "end time")
    dt = mo.ui.slider(0.01,2,0.01,value=0.6,show_value=True, label = "step size dt")
    x0 = mo.ui.slider(0,10,1,value=8,show_value=True, label = "initial condition x0")
    (tstop,dt,x0)
    return dt, tstop, x0


@app.cell(hide_code=True)
def __(dt, forward_euler_solve, ground_truth_vf, mo, plt, tstop, x0):
    # timepoints, xs = forward_euler_solve(vf, [1.0, 2, 3], 0, 10, 0.01)
    # plt.plot(timepoints, xs)  

    def plot_euler_vs_truth(x0,tstop,dt):


        timepoints, xs = forward_euler_solve(ground_truth_vf, [x0.value], 0, tstop.value, dt.value)

        fig, ax = plt.subplots()


        ax.plot(timepoints, xs, linewidth=3)

        # for i in range(len(x0)):
        #     ax.plot(timepoints, ground_truth_vf(x0[i], timepoints), label=f"True State {i+1}", linewidth=4)

        ax.set_xlabel('Time')
        ax.set_ylabel('State')
        return fig
    mo.as_html(plot_euler_vs_truth(x0,tstop,dt))
    return (plot_euler_vs_truth,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Initial observations

        - We can see that there are 'small enough' values for $dt$ where the Euler solution is pretty much perfect

        - As we increase $\delta t$, the approximation error seems to increase. 

        - At some point, there is a step change, and the Euler approximaton shows **qualitatively different behaviour**, rather than just being a bad approximation: the approximation error compounds.

        ### What's going on?!

        First let's note the true solution is monotonically *shrinking*:

        ---
        **True solution:** 

        $x(t) = x(0)\exp(-2.5t)$

        Therefore:

        $\Bigg| \frac{x(t + \delta t)}{x(t)} \Bigg| = \Bigg|\frac{\exp(-2.5(t+\delta t))}{\exp(-2.5t)} \Bigg|$

        $= | \exp(-2.5\delta t) | < 1$, for $\delta t > 0$.

        ---
        **Approximate solution:**

        $$\begin{align}
        x(t+ \delta t) 	&= (1 - 2.5 \delta t) x(t) \\
        				&= Ax(t)
        \end{align}$$

        So we can see that the solution is perfect if 

        $\exp(-2.5\delta t) = 1 - 2.5 \delta t$. 

        *If you're familiar with the Taylor expansion, ask yourself if there is something meaningful about this formula (see Enrico's extras a few cells below, not required for the exam!)*. Meanwhile:

        $\Bigg| \frac{x(t + \delta t)}{x(t)} \Bigg| = |A|.$

        Now if $|A| > 1$ we can see that the Euler solution will **grow** over time, instead of shrinking like the true solution. The solution is **unstable**.

        ---
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        !!! warning "Question"
            1. Use the maths I wrote down above to determine mathematically which values of $\delta t$ will make the Euler solution diverge (i.e. grow over time). Does your answer correspond to the results on the graph?
        
            #### Optional challenges (medium)
        
        
            2. Consider a single step of the Euler approximation for the differential equation we covered. So given $x(t)$, consider the value of $x(t + \delta t)$, for a stable value of $\delta t$. 
            * How much does it differ from the true solution? 
            * How does this approximation error decrease as we decrease the step size by a linear factor $k$?
            * How does the computational burden of simulation change as we decrease the step size?
            *Note: the Euler method doesn't get accurate very fast as we shrink the step size, compared to more sophisticated methods. Coupled with it's poor stability issues, it's not actually a great algorithm in practice*
        
            #### Optional challenges (hard)
            3. Suppose I give you an arbitrary value of $\delta t$. For instance, $\delta t = 0.01$, or more generally $\delta t = \epsilon$, for some $\epsilon$. Can you design a differential equation for which the Euler approximation, with $\delta t = \epsilon$, is unstable?
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.accordion({
        "Extra info by Enrico, not important for exam!" :
        mo.md(r"""
        ---

    We took the true solution to the linear ODE $\dot{x} = -ax(t)$ as $x(t) = x(0)\exp(-t)$ above. Where does this come from?

    A general solution can be found using the following steps, called **integration by separation of variables**.

    Assuming that we can split the fraction $\dot{x} = \mathrm{d}x / \mathrm{d}t$, we move the $\mathrm{d}t$ to the RHS and $x(t)$ to the LHS, then:

    $\int (1/x)\mathrm{d}{x} = \int -a \mathrm{d}{t} \Rightarrow \ln(x) = c - at \Rightarrow x(t) = e^{c}e^{-at} = x(0)e^{-at},$
    where the integration constant $e^{c} = x(0)$ is obtained by assuming that we know some initial condition $x(0)$ at time $t=0$. To see this explicitly, at time $t=0$ we have:

    $x(0) = e^{c}e^{-a0} \Rightarrow x(0) = e^{c}.$

    ---
        """)
    })
    return


@app.cell(hide_code=True)
def __(mo):
    mo.accordion(
        {"Question 1 answer" : 
    mo.md(r""" So we have seen that the true solution, $x(t) = x(0)e^{-2.5t}$, is monotonically shrinking: i.e., it is an exponential with a negative exponent: $-2.5\times t$, since $t>0$. Hence, $e^{-2.5t}$ will always be a number smaller than $1$ (to see this, note that you can write $e^{-2.5t} = \frac{1}{e^{2.5t}}$, if still unsure it is always a nice idea to plot it. **Crucially**: *we require our approximation to also satisfy this property*. Hence, we require that 

    $$
    \begin{align}
    \Bigg| \frac{x(t + \delta t)}{x(t)} \Bigg| &< 1 \\
    \Bigg| \frac{x(t + \delta t)}{x(t)} \Bigg| &= \Bigg| \frac{(1-2.5\delta{t})x(t)}{x(t)} \Bigg| \\
    & = |1-2.5\delta{t}| < 1
    \end{align}
    $$

    What $\delta{t}$ satisfies this inequality?

    $$\begin{align}
    &|1-2.5\delta{t}| < 1 \\
    \\
    &\Rightarrow1-2.5\delta{t} < 1\quad\text{or}\quad -1+2.5\delta{t} < 1 \\
    &\Rightarrow \delta{t} < \frac{2}{2.5} \approx 0.8
    \end{align}$$
    while the other inequality is satisfied for any $\delta{t}$.

    Of course, we only consider solutions for $\delta{t} > 0$, so the final answer is $\delta{t} < 0.8$. You can now try to go back to the plot above and slowly change $\delta{t}$ from $0.7$ to $0.81$. At exactly $0.8$ the Euler solution will start to oscillate with a constant period, while for values of $\delta{t} > 0.8$ it will exponentially increase!
    """)
                 })
    return


@app.cell(hide_code=True)
def __(mo):
    mo.accordion(
        {"Question 2 answer" : 
    mo.md(r""" 
    Use the code below to see how we can code an answer for this, you can use the Slider to change the value of the linear factor $k$. Below as an extra not required for the exam, you can also take a look at how we can answer to this question analytically.

    Finally, to answer to the last part of question 2, notice how the smaller the step size, the longer the computation. For instance, if we want to solve a differential equation for a particularly long time range, say between times $t_0 = 0$ seconds and $t_f = 10000$ seconds, then a step size of $\delta{t} = 0.01$ will already result in 

    $$\frac{t_f - t_0}{\delta{t}} = 10^6\quad\text{operations!}$$
    In this particular case Python has no issues performing $10^6$ operations, but if the Euler method calculation was a bit more complicated (for instance the solution of a system of equations, which will look like a matrix), this may be a big issue. Hence, it is important to be pragmatic sometimes! If we are computing some differential equation to calculate something like the speed of a car, we probably don't really care about incredible precisions unless you are a huge fan of the F1. Hence, in many cases an error of $\sim 0.02$ is more than acceptable.

    ---

    ---
    **Again, extra by Enrico: no need to know about this for the exam**

    Analytically, we may use the Taylor expansion of the true solution of the differential equation to see how the Euler step compares to the true solution.
    This is a bit more advanced but the [Taylor expansion](https://en.wikipedia.org/wiki/Taylor_series) is a method to rewrite a function as an infinite sum. For now, you may just convince yourself that this infinite sum is exactly the same as the original function!

    In practice, notice that the Taylor expansion of the exponential is:
    $$\exp(-2.5\delta t) = \sum_{k}\frac{(-2.5\delta{t})^{k}}{k!} = 1 - 2.5\delta{t} + \frac{1}{2!}(-2.5\delta{t})^2 + \frac{1}{3!}(-2.5\delta{t})^3 + \dots$$
    Note how the first two terms of the expansion of the RHS are exactly the same as our Euler step! i.e., $1 - 2.5\delta{t}$.

    Then, the rest of the infinite sum is basically the difference between our finite difference method and the true solution. Notice how the terms of the infinite series are some powers of $\delta{t}$. Since $\delta{t} < 1$, each will be smaller and smaller. For instance, using $\delta{t} = 0.1$ we have $\frac{1}{2!}(-2.5\delta{t})^2 = 0.03125$ while $\frac{1}{3!}(-2.5\delta{t})^3 = -0.0026041$ and so on, smaller and smaller.

    Hence, the highest error terms: $\frac{1}{2!}(-2.5\delta{t})^2 + \frac{1}{3!}(-2.5\delta{t})^3 = 0.02864$ will give us a good idea of the error of the Euler's method! indeed, using the code below, for $k=1$, we have $\delta{t} = 0.1$ and the error is around $0.028$.

    Plugging in the linear factor $k$ in the error terms from the Tayor example will give us an idea of how the solution will change if we had a different $\delta{t}$.
    """)
                 })
    return


@app.cell(hide_code=True)
def __(mo):
    mo.accordion(
        {"Question 3 answer" : 
    mo.md(r""" 
    One way to do this, is to "reverse-engineer" our solution to question 1. Consider the general linear ordinary differential equation $\dot{x} = -a x(t)$ with solution $x(t) = x(0)e^{-at}$ for $a > 0$.

    Then, our approximate solution using the Euler step will look like: $x(t+ \delta t)	= (1 - a \delta t) x(t)$. Similarly to question 1, we require $|(1 - a \delta t)| < 1$ for stable solutions and $|(1 - a \delta t)| > 1$ for unstable solutions. Then, for unstable solutions, assuming that $\delta{t} = \epsilon$ we have:

    $$\begin{align}
    |(1 - a \epsilon)| &> 1 \\
    (1 - a \epsilon) > 1\quad &\text{or}\quad -(1 - a \epsilon) > 1
    \end{align}$$

    the first inequality is true for any $\epsilon > 0$ since we assumed that $a>0$. The second inequality instead is true as long as $a\epsilon > 2$. By plugging in $\delta{t} = \epsilon = 0.01$ we find $a > 2 / \epsilon = 200$.

    Hence, if we had the differential equation $\dot{x} = -200 x(t)$ the Euler step solution will be unstable for $\delta{t}  = 0.01$. You can check this by changing the `ground_truth_vf` and `plot_euler_vs_truth` functions from a few cells above, using $-200$ rather than $-2.5$ in the exponent.
    """)
                 })
    return


@app.cell
def __(mo):
    kvals = [1,10,25,50,100,500,1000,10000]
    kvals
    k = mo.ui.slider(steps=list(kvals), show_value = True, label = "linear factor k")
    k
    return k, kvals


@app.cell
def __(k):
    k.value
    return


@app.cell
def __(k, math, np):

    def give_true_solution(k):
        def true_solution(t):
            return np.exp(-2.5 * t)
        
        def error_terms(dt, power):
            return (1 / math.factorial(power)) * (-2.5 * dt) ** power

        dt = 0.1 / k.value
        
        euler_step = 1 * (1 - 2.5 * dt)
        true_sol = true_solution(dt)

        print(f"Question 2\n")
        print(f"Euler's method after one step using x(0) = 1: {euler_step}")
        print(f"True solution after time t = Δt using x(0) = 1: {true_sol}")
        
        error = abs(euler_step - true_sol)
        print(f"\nError or difference between Euler's method and true solution: {error}\n using Δt = {dt} and k = {k.value}.")
        print("Question 2 Extra by Enrico\n")
        print(f"Error from first two error terms in the expansion:\n{sum([error_terms(dt, i) for i in range(2, 4)])}")

        print(f"\nError or difference between Euler's method and true solution:\n{abs(1 - 2.5 * dt - true_sol)}")
    give_true_solution(k)
    return (give_true_solution,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Final notes on Forward Euler

        - There are literally hundreds of algorithms like Forward Euler for numerically solving ODEs. Forward Euler is the most basic. You can use an ODE solving package 

        - In real life, you would not code up the numerical algorithm (Forward Euler/...) yourself. You would use a package like [this](https://docs.sciml.ai/OrdinaryDiffEq/stable/) (docs [here](https://docs.sciml.ai/DiffEqDocs/stable/getting_started/)). 

        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Full SIR Model

        Let's get back to modelling pandemics. Recall that 

        -  $S$ is susceptible (healthy, previously uninfected people)
        -  $I$ is currently infected people
        -  $R$ is recovered people

        From the lecture, we had 

        $$\dot{S}(t) = -p\frac{S(t)I(t)}{N}$$

        $$\dot{I}(t) = p\frac{S(t)I(t)}{N} - \gamma I(t)$$

        $$\dot{R}(t) = \gamma I(t)$$

        !!! warning "Question"
            List as many assumptions as you can on the SIR model above
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.accordion(
        {"Answer" : 
    mo.md(r""" 
    - Infections follow a binomial distribution with probability equal to infection rate $p$.
    - Infections are proportional to the proportion of the infected population.
    - Recoveries follow a binomial distribution with probability equal to recovery rate $γ$.
    - Recovered people do not become susceptible again.
    """)
                 })
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Let's code up the vector field above (see `build_SIR_dynamics`) below. Instant issue, the vector field depends on hyperparameters. But for `forward_euler_solve`, it should only depend on states $x$ and time $t$. How can we fix this issue?

        - One way (see below), is to have a 'factory' function, that takes in hyperparameters, and emits the actual function, which is now seeded with the appropriate hyperparameters. 

        - Note that some hyperparameters are shared with the SI model above: you can slide them there. If this is annoying, open the notebook in two windows simultaneously!
        """
    )
    return


@app.cell
def __(np):
    def build_SIR_dynamics(hyps):
        p, γ, N = hyps["p"], hyps["γ"], hyps["N"]
        
        def SIR_dynamics(x, t):
            S, I, R = x
            return np.array([-p * S * I / N, p * S * I / N - γ * I, γ * I])
        
        return SIR_dynamics

    return (build_SIR_dynamics,)


@app.cell
def __(mo, np):
    n_SIR_students = mo.ui.slider(5,200,5, show_value=True, value = 100, label = "number of students")
    euler_steps = 10**np.linspace(-4,2,7)
    return euler_steps, n_SIR_students


@app.cell
def __(euler_steps, mo, n_SIR_students):
    initial_SIR_infections = mo.ui.slider(5,n_SIR_students.value,5, show_value=True, value = 10, label = "initial infections")
    SIR_days = mo.ui.slider(5,200,5, show_value=True, value = 50, label = "number of days to simulate")
    recovery_rate = mo.ui.slider(0,1,0.01, show_value=True, value = 0.1, label = "recovery rate")
    infectiveness = mo.ui.slider(0,1,0.01, show_value=True, value = 0.5, label = "infectivity rate")
    euler_step = mo.ui.slider(steps=list(euler_steps), show_value=True, value = 0.01, label ="step size for forward euler")

    (n_SIR_students, initial_SIR_infections, SIR_days, recovery_rate, infectiveness, euler_step)
    return (
        SIR_days,
        euler_step,
        infectiveness,
        initial_SIR_infections,
        recovery_rate,
    )


@app.cell
def __(
    SIR_days,
    euler_step,
    infectiveness,
    initial_SIR_infections,
    n_SIR_students,
    recovery_rate,
):
    SIR_hyperparameters = {
        "N": n_SIR_students.value,
        "I₀": initial_SIR_infections.value,
        "R₀": infectiveness.value/recovery_rate.value,  # initial recovered
        "days": SIR_days.value,
        "p": infectiveness.value,
        "γ": recovery_rate.value,
        "δt": euler_step.value
    }
    return (SIR_hyperparameters,)


@app.cell
def __(
    SIR_hyperparameters,
    build_SIR_dynamics,
    forward_euler_solve,
    mo,
    np,
    plt,
):
    def build_initial_conditions(hyperparameters):
        S0 = hyperparameters["N"] - hyperparameters["I₀"] - hyperparameters["R₀"]
        I0, R0 = hyperparameters["I₀"], hyperparameters["R₀"]
        return np.array([S0, I0, R0])

    def plot_SIR(hyps):
        ts, xs = forward_euler_solve(
            build_SIR_dynamics(hyps), 
            build_initial_conditions(hyps),
            0,
            hyps["days"],
            hyps["δt"]
        )
        
        fig, ax = plt.subplots()
        ax.plot(ts, xs, label = ["susceptible","infected","recovered"])    

        
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Population")
        ax.legend()
        
        return fig
    mo.as_html(plot_SIR(SIR_hyperparameters))
    return build_initial_conditions, plot_SIR


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        !!! warning "Question"
            1. Go through the lecture notes and make sure you understand $R_0$, and why $\dot{I}(t) < 0$ always if $R_0 < 1$. 
        
        
            2. Make a function whose input is the hyperparameters, and whose output is $R_0$. Run the function below the SIR plot, so you can see how changing the parameters changes both the $R_0$, and the dynamics of the pandemic
        
            3. Suppose I extended the model by allowing recovered people to eventually become susceptible again (i.e. they lose immunity). Would this affect the statement that $R_0 < 1$ avoids a pandemic?
        
        
            **Optional challenge**
            - Extend the model by allowing recovered people to eventually become susceptible again (i.e. they lose immunity)
            - Add seasonality to the model. EG the infectiveness varies sinusoidally with some frequency.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.accordion(
        {"Answer" : 
    mo.md(r""" 
    1. If at time point $t$ we have a population of $I(t)$ infected people and on average each one of them infects other $R_0$ people, then at time $t+1$ the population will become a group of $\dot{R}(t+1)=I(t)$ recovered people and $I(t+1)=I(t)*R_0$ infected people. The rate of change of infected population is $\dot{I}(t) = I(t+1)-I(t) = I(t)*R_0-I(t) = I(t)*(R_0-1)$. We can then see that if $R_0=1$ the number of infected people won't change ($\dot{I}=0$), if $R_0>1$, the number of infected people will keep rising ($\dot{I}>0$), and if $R_0<1$, the number of infected people will keep falling ($\dot{I}<0$). 
     
    3. Allowing recovered people to become susceptible again returns some of the population back to the susceptible pool, but it doesn't change the infection dynamics. If $R_0<1$, then $\dot{I}<0$, which means that the number of infections will keep falling regardless of how many susceptible people there are.
    """)
                 })
    return


@app.cell
def __(SIR_hyperparameters):
    def calculate_r0(h):
        return h["p"] / h["γ"]
    calculate_r0(SIR_hyperparameters)
    return (calculate_r0,)


@app.cell(hide_code=True)
def __(mo):
    mo.accordion(
        {"Answers to optional challenge" : 
    mo.md(r""" You could first make an extended dictionary with the extra parameters like this:

    ```SIR_hyperparameters_reinfection = {
        "N": n_SIR_students_new,
        "I₀": initial_SIR_infections_new,
        "R₀": 0,  # initial recovered
        "days": SIR_days_new,
        "p": infectiveness_new,
        "γ": recovery_rate_new,
        "r": resusceptibility_rate,
        "δt": euler_step,
        "freq": seasonality_frequency,
        "amp": seasonality_severity
    }
    ```
    Then you could make a new vector field for the modified dynamics. EG this:

    ```
    def build_SIR_dynamics_reinfection(hyps):
        p, γ, N, r, freq, amp = [hyps[el] for el in ["p", "γ", "N", "r", "freq", "amp"]]
        
        def SIR_dynamics(x, t):
            S, I, R = x
            seasonality_factor = np.sin(t * freq) * amp + 1
            dS = r * R - p * S * I / N * seasonality_factor
            dI = p * S * I / N * seasonality_factor - γ * I
            dR = γ * I - r * R
            return [dS, dI, dR]
        
        return SIR_dynamics
    ```

    """)
                 })
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Full stochastic SIR model

        - Our ODE above was a mean field approximation, as derived in the lectures
        - Here we simulate the full stochastic solution

        """
    )
    return


@app.cell
def __(SIR_hyperparameters, np):
    def run_stochastic_simulation(hyps):
        days = hyps["days"]
        N = hyps["N"]
        SIR = np.zeros((3, days))
        I0, R0 = hyps["I₀"], hyps["R₀"]
        S0 = hyps["N"] - I0 - R0
        p = hyps["p"]
        γ = hyps["γ"]
        
        # Initialize the SIR array with the initial conditions
        SIR[:, 0] = [S0, I0, R0]

        # Define the infection and recovery functions
        def infections(i):
            return np.random.binomial(SIR[0, i-1], p * SIR[1, i-1] / N)

        def recoveries(i):
            return np.random.binomial(SIR[1, i-1], γ)

        # Update function for SIR
        def SIR_update(i):
            S, I, R = SIR[:, i-1]
            inf = infections(i)
            rec = recoveries(i)
            S = S - inf
            I = I + inf - rec
            R = R + rec
            return [S, I, R]

        # Run the simulation over the specified days
        for i in range(1, days):
            SIR[:, i] = SIR_update(i)  # Update the SIR values for each day
        
        return SIR

    run_stochastic_simulation(SIR_hyperparameters)
    return (run_stochastic_simulation,)


@app.cell
def __(SIR_hyperparameters, np, plot_SIR, plt, run_stochastic_simulation):
    def plot_stochastic_and_ODE_SIR(hyps):
        # Plot the deterministic SIR
        fig = plot_SIR(hyps)
        ax = fig.gca()
        # Run the stochastic simulation
        SIR_stoch = run_stochastic_simulation(hyps)
        
        # Scatter plot the stochastic results
        ax.scatter(np.arange(hyps["days"]), SIR_stoch[0, :], color="blue", label="susceptible (stochastic)", alpha=0.5)
        ax.scatter(np.arange(hyps["days"]), SIR_stoch[1, :], color="red", label="infected (stochastic)", alpha=0.5)
        ax.scatter(np.arange(hyps["days"]), SIR_stoch[2, :], color="green", label="recovered (stochastic)", alpha=0.5)
        
        plt.show()
    plot_stochastic_and_ODE_SIR(SIR_hyperparameters)
    return (plot_stochastic_and_ODE_SIR,)


if __name__ == "__main__":
    app.run()
