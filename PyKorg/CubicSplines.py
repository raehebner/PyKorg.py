import numpy as np


# Message from Korg's CubicSplines.py:

# This module contains modified functions from DataInterpolations.jl (license below).
# We aren't depending on DataInterpolations because its dependencies are very heavy.
# In the future, hopefully we can eliminate interpolation code and depend on a widely-used
# well-tested and lightweight library.

# Note, I've swapped the order of t and u (t is the abscissae/x-values, u are the y-values). The other 
# major change is that I've simplified the types. We might at some point want to add analytic 
# derivatives (DataInterpolations has these), but since ForwardDiff doesn't use ChainRules, there is 
# no way to get them used in autodiff (until/unless you use a different library.)

# Copyright (c) 2018: University of Maryland, Center for Translational Medicine.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Note that the typing is somewhat more lasez-faire in python as opposed to julia

class CubicSpline:
	def __init__(self,t,u,h,z,extrapolate):
		self.t = t
		self.u = u
		self.h = h
		self.z = z
		self.extrapolate = extrapolate

	def show(self):
		print("t:",self.t)
		print("u:",self.u)
		print("h:",self.h)
		print("z:",self.z)
		print("extrapolate:",self.extrapolate)

	def interpolate(self,t):
		"""
		CubicSpline.interpolate(x)

	    Returns an interpolated value using the Cubic Spline at any x value in the domain.

		If `CubicSpline.extrapolate` is false, x values outside [`xs[1]`, `xs[end]`] throw errors, if `CubicSpline.extrapolate` is
		true, the interpolant uses flat extrapolation, i.e. it returns the extreme value.
		"""
		if not(self.t[0] <= t <= self.t[-1]):
			if self.extrapolate:
				if t < self.t[0]:
					return self.u[0]
				else:
					return self.u[-1]
			else:
				raise ValueError(f"Out-of-bounds value {t} passed to interpolant. Must be between {A.t[1]} and {A.t[end]}")

		i = max(0,min(np.searchsorted(self.t,t),len(self.t)-2))
		I = self.z[i] * (self.t[i+1] - t)**3 / (6*self.h[i+1]) + self.z[i+1] * (t - self.t[i])**3 / (6*self.h[i+1])
		C = (self.u[i+1] / self.h[i+1] - self.z[i+1] * self.h[i+1] / 6) * (t - self.t[i])
		D = (self.u[i] / self.h[i+1] - self.z[i] * self.h[i+1] / 6) * (self.t[i+1] - t)
		return I + C + D

	def cumulative_integral(self,t1,t2):
		"""
    	CubicSpline.cumulative_integral(t1, t2)

		Given a curve described by the spline, calculates the integral from t1 to t2 for all t = t1, t2, and
		all spline knots in between.  So if `t1` is `CubicSpline.t[0]` and `t2` is `CubicSpline.t[-1]`, `out` should have the
		same length as `CubicSpline.t`.
		"""
		idx1 = max(0,min(np.searchsorted(self.t,t1),len(self.t)-2))
		idx2 = max(1,min(np.searchsorted(self.t,t2),len(self.t)-2))
		if self.t[idx2] == t2:
			idx2 -= t1

		out = np.zeros(len(self.t),dtype=type(self.u))
		for idx in range(idx1,idx2+1):
			lt1 = t1 if idx == idx1 else self.t[idx]
			lt2 = t2 if idx == idx2 else self.t[idx+1]
			out[idx+1] = out[idx] + self._integral(idx,lt2) - self._integral(idx,lt1)
		out = [x for x in out if x != 0]
		return out

	def _integral(self,idx,t):
		t1 = self.t[idx]
		t2 = self.t[idx+1]
		u1 = self.u[idx]
		u2 = self.u[idx+1]
		z1 = self.z[idx]
		z2 = self.z[idx+1]
		h2 = self.h[idx+1]
		return (t**4 * (-z1 + z2) / (24 * h2) + t**3 * (-t1 * z2 + t2 * z1) / (6 * h2) +
		 t**2 * (h2**2 * z1 - h2**2 * z2 + 3 * t1**2 * z2 - 3 * t2**2 * z1 - 6 * u1 + 6 * u2) / (12 * h2) +
		 t * (h2**2 * t1 * z2 - h2**2 * t2 * z1 - t1**3 * z2 - 6 * t1 * u2 + t2**3 * z1 + 6 * t2 * u1) /
		 (6 * h2))



def construct_spline(t,u,extrapolate=False):
	"""
	construct_spline(xs, ys, extrapolate=false)

	Construct a interpolant using `xs` and `ys` as the knot coordinates. Assumes `xs` is sorted. 

	Returns a `CubicSpline` object, which can be called with `CubicSpline.interpolate(x)` to interpolate at 
	any x value in the domain. 

	If `extrapolate` is false, x values outside [`xs[1]`, `xs[end]`] throw errors, if `extrapolate` is
	true, the interpolant uses flat extrapolation, i.e. it returns the extreme value.
	"""
	n = len(t) - 1
	h = [0,*map(lambda k: t[k+1] - t[k],range(len(t)-1)),0]
	dl = h[1:n+1]
	d_tmp = [2 * (i + j) for i,j in zip(h[0:n+1],h[1:n+2])]
	du = h[1:n+1]
	tA = np.diag(dl, -1) + np.diag(d_tmp, 0) + np.diag(du, 1)
	d = [0,*[6*(u[i]-u[i-1]) / h[i] - 6*(u[i-1]-u[i-2]) / h[i] for i in range(1,n)],0]
	z = np.linalg.solve(tA,d)
	return CubicSpline(t,u,h[0:n+1],z,extrapolate)

