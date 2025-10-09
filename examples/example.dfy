
module Example {
  function plus(x: nat, y: nat): nat {
    x + y
  }

  lemma AddCommut(n: nat, m: nat)
    ensures plus(n,m) == plus(m,n)
  {
    // intentionally empty, will fail and be sketched
  }
}
