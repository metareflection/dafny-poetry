
module Example {
  function sum(n: nat): nat
    decreases n
  {
    if n == 0 then 0 else n + sum(n-1)
  }

  lemma {:induction false} SumFormula(n: nat)
    ensures 2 * sum(n) == n * (n + 1)
  {
    // intentionally empty, will fail without induction
  }
}
