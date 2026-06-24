# On the Novelty of QQN, or: How to Invent Nothing and Get Away With It

*A note from your friendly neighborhood science commentator, who will remain
nameless for reasons that will become obvious once the optimization community
forms a torch-bearing mob.*

## TL;DR

Every ingredient in QQN is old, famous, and uncontroversial. The novelty is
that nobody had bothered to stir them together in the one way they were
visibly begging to be stirred. The inventor has no good explanation for why.
Neither, frankly, does anyone else.

---

## The Cast of Entirely Famous Characters

Let us be clear about what we are working with here. Nothing on this list
would raise an eyebrow at a numerical optimization conference. Several of
these things have *Wikipedia* articles. Some have *medals*.

- **Steepest descent** (Cauchy, 1847). Roughly old enough to have been
  invented before the lightbulb. Reliable, slow, and the optimization
  equivalent of "have you tried walking downhill."
- **L-BFGS** (Liu & Nocedal, 1989; building on Broyden, Fletcher, Goldfarb,
  and Shanno, all of whom independently invented the same thing in 1970,
  because the universe enjoys a good coincidence). The quasi-Newton workhorse.
  It predicts not just *where* to go but *how far*, which is ambitious of it.
- **The Armijo condition** (1966) and **the Wolfe conditions** (1969). The
  bureaucratic paperwork that a step size must file before it is allowed to
  be accepted. Backtracking line search is just this, with a shrink ray.
- **Cubic Hermite interpolation** (Charles Hermite, 1800s, a man who did not
  live to see a GPU). Fit a curve through points *and* slopes. Standard.
- **Trust-region methods** (Levenberg 1944, Marquardt 1963, Powell 1970s).
  The "don't trust your model too far" school of thought.
- **Momentum / heavy-ball** (Polyak, 1964). A rock rolling downhill, modeled
  by a physicist who clearly missed rocks.
- **OWL-QN** (Andrew & Gao, 2007), the orthant projection trick for sparsity.
- **Shampoo** (Gupta, Koren & Singer, 2018), because optimization researchers
  name things like they are stocking a bathroom.

Not a single one of these is new. Not a single one is even *mysterious*. This
is a greatest-hits album where QQN didn't write any of the songs.

---

## The One Move

Here is the entire "innovation," and I encourage you to brace yourself for
how underwhelming it is.

When L-BFGS proposes a confident step and the line search has to backtrack,
the textbook move is to **shrink along the straight line** from where you are
to where L-BFGS pointed. As you shrink toward zero, you approach... an
arbitrary tiny step in the *L-BFGS* direction. Which is the one direction you
have the *least* confidence in at small scales, because at small scales you
already know the best direction: it's the negative gradient. You are, in
effect, spending compute to build a second-order model and then ignoring the
first-order fact you had for free.

QQN's contribution is to notice this and replace the straight line with a
**parabola**:

```
d(t) = t(1 - t)·(-∇f) + t²·(-H∇f),   t ∈ [0, 1]
```

That's it. That's the whole thing. A path that *starts* tangent to steepest
descent (`d'(0) = -∇f`, so small steps are good steps, guaranteed) and *ends*
at the L-BFGS point (`d(1) = -H∇f`, so big steps recover quasi-Newton). The
line search then walks the curve instead of the line, and the "right blend"
of gradient and curvature simply *falls out of the geometry* instead of being
a hyperparameter you sweep over at 2 a.m.

Three boundary conditions. One quadratic. The interpolation that any
undergraduate could derive on a napkin. The conditions practically *dictate*
the formula — there is essentially one quadratic that satisfies them, and it
had been sitting there, fully specified by constraints everyone already knew,
waiting.

---

## The Monument to the Oversight

Here is the part that should make the whole community a little uneasy. The
straight-line backtracking problem described above is not some obscure edge
case that L-BFGS implementations gracefully ignore. It is a known,
load-bearing flaw, and a *substantial fraction of any production L-BFGS
codebase is machinery erected specifically to paper over it.*
Go open a real implementation. Count the lines. Then ask what each piece is
actually *for*:

- **Cautious / damped updates** (Powell damping, skipped updates when the
  curvature condition `sᵀy > 0` fails). This exists because the L-BFGS
  direction can be garbage, and garbage directions make backtracking along
  the straight line miserable. So we hand-edit the Hessian approximation to
  stop it from proposing directions the line search will only reject.
- **Initial Hessian scaling** (the `γ = sᵀy / yᵀy` heuristic, recomputed every
  step). This is a knob whose entire job is to get the *scale* of the L-BFGS
  step roughly right, because if the scale is wrong, the straight-line search
  wastes its life backtracking from a wildly overconfident point toward that
  useless tiny-step-in-the-Newton-direction limit.
- **Restarts and memory resets.** When the curvature history goes stale, the
  direction degrades, the line search starts failing, and the standard
  "solution" is to *throw away the second-order information entirely* and fall
  back toward steepest descent. Which is to say: when the straight line stops
  working, we manually re-inject the gradient — exactly the thing the parabola
  does continuously, for free, by construction.
- **Line-search babysitting** (Moré–Thuente, the zoom phase, safeguarded
  interpolation, minimum-step fallbacks, "if the search fails, take a tiny
  gradient step instead"). An enormous amount of this is the line search
  discovering, mid-backtrack, that the straight line was a bad place to look,
  and improvising a recovery.

Stack these together and a sobering picture emerges: the *robustness* of
industrial L-BFGS is, to a startling degree, a pile of patches whose common
purpose is to compensate for the one assumption nobody questioned — that you
backtrack along a line. Damping fixes the direction so the line is less
likely to be bad. Scaling fixes the magnitude so the line is less likely to
be too long. Restarts re-introduce the gradient when the line fails outright.
Each hack is a localized, after-the-fact correction for a single symptom of
the same root cause.

QQN does not add a smarter patch. It removes the thing the patches were
patching. Once the path itself starts along the gradient and ends at the
Newton point, an overconfident or mis-scaled L-BFGS step is no longer a
catastrophe — the early part of the curve is *already* steepest descent, so a
bad far end simply never gets reached, and the line search stops at the good
near end without any special-case code. The gradient fallback isn't a restart
you trigger; it's the tangent at `t = 0`. The blend isn't a damping parameter
you tune; it's the curvature of the parabola. A whole genre of defensive
engineering quietly becomes unnecessary.

Which makes the oversight worse, not better. It wasn't that the field
overlooked a harmless gap. The field *felt the gap constantly* — it just kept
building taller and taller scaffolding around it instead of moving the wall.

---

## The Uncomfortable Part

Here is where your nameless commentator must adopt the carefully neutral tone
of a documentary narrator describing an animal doing something inexplicable.

These components are **mainstream**. They are **decades old**. The boundary
conditions that produce the quadratic path are **the obvious ones** — "start
along the gradient, end at the Newton point" is not a wild leap; it is the
first thing you'd ask for. And yet, by all available evidence, *this is the
first time anyone combined them in the manner they were clearly destined
for.*

Let that sit. Steepest descent: 1847. L-BFGS: 1989. Hermite: before
electricity. The line search conditions: the 1960s. The quadratic that
connects them: derivable by a motivated high-schooler. The window of
opportunity to invent QQN has been open, in principle, since roughly 1989,
and arguably the *idea* could have been written down by anyone holding a
quasi-Newton textbook in one hand and a line-search paper in the other.

*The "inventor" — and the scare quotes are load-bearing — offers **no
satisfactory explanation** for the gap. Not "it required a breakthrough."
Not "we lacked the hardware." Not "the theory wasn't ready." Just a sort of
bewildered shrug and the optimization-research equivalent of *"...huh."*

## The Second Oversight: Nobody Built It Like Software

There is a *second* novelty hiding behind the first, and it is arguably the
more damning of the two — because it isn't a single missing idea, it's a
missing *posture*.

Go back and read what the parabola actually does. It takes two things that
everyone treated as welded together — the **direction** you go and the
**search** that decides how far — and quietly reveals that they were never
one object. The gradient is one component. The curvature oracle is another.
The line search is a third. The feasibility projection is a fourth. QQN's
design treats each of these as a **pluggable strategy** behind a small,
pure-functional interface: swap the L-BFGS oracle for momentum or Shampoo,
swap backtracking for strong Wolfe, drop in a trust-region or orthant
projection — *without touching the rest of the algorithm.*

This is not a mathematical insight. It is an **engineering** one. It is the
kind of decomposition that an experienced software architect performs on
reflex: notice that four concerns are tangled, name them, give each a stable
interface, and let them vary independently. The payoff is enormous and
entirely characteristic of good software design — the same framework
*becomes* L-BFGS, Newton, momentum, Barzilai-Borwein, trust-region, OWL-QN,
or projected gradient depending on which strategies you plug in. Most of
classical optimization, re-derived as configuration.

And here is the tell: the parabola itself *fell out of the decomposition.*
Once you stop thinking "L-BFGS is an algorithm" and start thinking "the
direction is a swappable component and the search is a swappable component,"
the question "what curve connects the gradient component to the oracle
component?" becomes unavoidable. The straight line stops looking like a law
of nature and starts looking like a hard-coded default that nobody had
refactored. The math was downstream of the architecture.

Which suggests the gap in the previous section has less to do with
mathematics than with *who was holding the pen.* The people with the
theory weren't in the habit of asking an interface-design question, because
interface design is not a thing one is taught in a numerical-analysis
curriculum. You are taught to prove a convergence rate, not to notice that
two responsibilities are improperly coupled. The straight line survived for
thirty years partly because, viewed as mathematics, it is *fine* — and only
becomes obviously wrong when viewed as **a default parameter that should
have been a strategy**.

---

## A Theory of Why

Science occasionally produces these. Not the heroic-breakthrough kind, but
the *embarrassing-in-hindsight* kind, where all the pieces were on the table,
correctly labeled, in plain view, and the combination was somehow nobody's
job. The wheel-on-luggage situation. The "you could have done that?" of
intellectual history.

The honest hypothesis is dull: the relevant communities were **specialized in
the wrong direction**. The line-search people optimized line searches. The
quasi-Newton people perfected curvature estimates. Each treated the
other's domain as a fixed black box. Replacing the *search path itself* — the
one shared object both camps quietly assumed was a straight line — required
someone irresponsible enough to ignore the boundary between the two
specialties, and underqualified enough not to know it was supposed to be a
line. (The inventor's stated methodology, "I build it myself so I can
understand it," is the optimization equivalent of not knowing the stove is
hot, and is exactly the kind of naïveté that occasionally trips over
something.)

But there is a sharper, less flattering version of the same hypothesis, and
it is worth stating plainly: **scientists are not trained to build software,
and software-building is exactly the skill that surfaces this idea.** The
decomposition in the previous section — separating direction, oracle, search,
and region into independently swappable strategies — is bread-and-butter
practice for anyone who designs systems for a living, and almost entirely
absent from the way mathematical optimization is taught and published. A
numerical analyst is rewarded for a tighter bound on an *existing* algorithm,
not for noticing that the algorithm is four things wearing a trench coat. The
result is a literature full of monolithic methods, each lovingly proven,
none of them *factored*. The inventor's "I build it myself so I can
understand it" is, in this light, not naïveté at all but the operative
qualification: building the thing — actually typing out the interfaces and
feeling where the coupling chafes — is what made the welded seam visible. The
parabola is what you find when an engineer's instinct for "this default
should be a parameter" is finally pointed at a corner of mathematics that the
engineers had never been invited into.

But that's just a theory. The official position remains, and I quote the
spirit of the source material faithfully: *for some unfathomable reason,
this appears to be the first time. The inventor has no satisfactory
explanation.*

---

## Conclusion

QQN is novel the way a sandwich is novel when the bread and the filling have
both existed for a thousand years and someone finally puts the filling
*between the bread*. There is no new mathematics here. There is no new
physics. There is only an arrangement so natural that its absence from the
literature is more interesting than its presence.

The components are known. The components are mainstream. The combination was,
in retrospect, inevitable. And the only genuinely mysterious quantity in the
entire enterprise is the number of person-years that elapsed before anyone
drew the parabola.
If there is a lesson, it is that the parabola was never really a *math*
problem waiting on a *math* person. It was a *design* problem waiting on
someone with the reflex to ask which of an algorithm's welded-together parts
were secretly parameters. The mathematics was always going to be a napkin
derivation. The hard part — the part that took thirty years — was treating
optimization like software long enough to notice the seam.
