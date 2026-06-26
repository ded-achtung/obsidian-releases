# A Short Primer of Pure Mathematics

This is a sample mathematical text used to test whether the system learns from
real material. Drop your own books (PDF or Markdown) into this folder and the
pipeline will read them too.

## 1. Natural numbers and induction

The natural numbers are the numbers 0, 1, 2, 3, and so on. They satisfy the
principle of mathematical induction: if a statement holds for 0, and whenever it
holds for a natural number n it also holds for n + 1, then it holds for every
natural number.

Theorem. For every natural number n, the sum of the first n positive integers
equals n times (n + 1) divided by 2.

Proof. We argue by induction on n. For the base case n = 0 the sum is empty and
equals 0, which agrees with the formula. Assume the statement holds for some
natural number n, so that 1 + 2 + ... + n equals n(n + 1)/2. Then the sum of the
first n + 1 integers equals n(n + 1)/2 plus (n + 1), which equals (n + 1)(n + 2)/2.
This is exactly the formula for n + 1, so by induction the statement holds for
every natural number. This completes the proof.

## 2. Divisibility and prime numbers

Let a and b be integers. We say that a divides b, and write a | b, if there is an
integer c such that b equals a times c. A prime number is an integer p greater
than 1 whose only positive divisors are 1 and p itself. An integer greater than 1
that is not prime is called composite.

Theorem (Euclid). There are infinitely many prime numbers.

Proof. Suppose, for the sake of contradiction, that there were only finitely many
primes, say p_1, p_2, up to p_k. Consider the number N equal to the product of all
these primes plus 1. The number N is greater than 1, so it has at least one prime
divisor q. But q cannot be any of the primes p_1 through p_k, because dividing N by
any of them leaves a remainder of 1. Hence q is a prime not in our list, which
contradicts the assumption that the list contained every prime. Therefore there are
infinitely many primes.

The fundamental theorem of arithmetic states that every integer greater than 1 can
be written as a product of primes, and that this factorization is unique up to the
order of the factors. For example, the number 360 factors as 2 times 2 times 2
times 3 times 3 times 5.

## 3. Rational and real numbers

A rational number is a number that can be written as a fraction a over b, where a
and b are integers and b is not zero. The real numbers extend the rationals so that
every bounded set has a least upper bound. Not every real number is rational.

Theorem. The square root of 2 is irrational.

Proof. Suppose the square root of 2 were rational, equal to a over b in lowest
terms, so that a and b share no common factor greater than 1. Then a squared equals
2 times b squared, so a squared is even, hence a is even. Write a equal to 2 times c.
Then 4 times c squared equals 2 times b squared, so b squared equals 2 times c
squared, hence b is also even. But then a and b share the factor 2, contradicting
the assumption that the fraction was in lowest terms. Therefore the square root of 2
is irrational.

## 4. Limits and continuity

Let f be a function defined near a point a. We say that the limit of f at a is L if
for every positive number epsilon there is a positive number delta such that
whenever the distance from x to a is less than delta and x is not a, the distance
from f of x to L is less than epsilon. A function f is continuous at a if the limit
of f at a exists and equals f of a.

Continuous functions on a closed bounded interval attain their maximum and minimum
values. Moreover, a continuous function that takes a negative value and a positive
value on an interval must take the value zero somewhere between them; this is the
intermediate value theorem.

## 5. Derivatives

The derivative of a function f at a point a measures the instantaneous rate of
change of f at a. It is defined as the limit, as h tends to zero, of the quotient
of f of a plus h minus f of a, divided by h, when this limit exists. The derivative
of the function x squared is 2 times x. The derivative of a sum is the sum of the
derivatives, and the derivative of a product f times g equals f prime times g plus
f times g prime.

Theorem (mean value theorem). If f is continuous on the closed interval from a to b
and differentiable on the open interval, then there is a point c between a and b at
which the derivative of f equals the average rate of change, that is, f of b minus
f of a divided by b minus a.

## 6. Integrals

The integral of a non-negative continuous function over an interval measures the
area under its graph. The fundamental theorem of calculus connects differentiation
and integration: if F is an antiderivative of a continuous function f, then the
integral of f from a to b equals F of b minus F of a. Integration is linear, and
integration by parts follows from the product rule for derivatives.

## 7. Sequences and series

A sequence is an ordered list of numbers indexed by the natural numbers. A sequence
converges to a limit L if its terms get and stay arbitrarily close to L. A series is
the sum of the terms of a sequence. The geometric series with ratio r whose absolute
value is less than 1 converges to its first term divided by one minus r.

The harmonic series, the sum of the reciprocals of the positive integers, diverges,
even though its terms tend to zero. In contrast, the sum of the reciprocals of the
squares of the positive integers converges, and its value is pi squared divided by 6.

## 8. Vectors and matrices

A vector in n dimensions is an ordered list of n real numbers. Vectors are added
component by component and can be scaled by real numbers. The dot product of two
vectors is the sum of the products of their corresponding components; it is zero
exactly when the vectors are orthogonal.

A matrix is a rectangular array of numbers. Multiplying a matrix by a vector
produces a new vector, and matrix multiplication composes linear transformations.
A square matrix is invertible if and only if its determinant is not zero. The
eigenvalues of a square matrix are the numbers lambda for which there is a non-zero
vector v with A times v equal to lambda times v; such a vector is an eigenvector.

## 9. Groups

A group is a set together with an operation that combines two elements to form a
third, satisfying three rules: the operation is associative, there is an identity
element, and every element has an inverse. The integers under addition form a group,
with identity 0 and the inverse of n equal to minus n. The symmetries of a square
form a group with eight elements, called the dihedral group of order eight.

Theorem (Lagrange). In a finite group, the number of elements of any subgroup
divides the number of elements of the whole group.

These ideas, from induction through groups, form the common language of pure
mathematics: precise definitions, careful statements, and proofs that proceed by
logical steps from assumptions to conclusions.
