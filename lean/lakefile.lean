import Lake
open Lake DSL

package «lambda-opt» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «LambdaOpt» where
  roots := #[`LambdaOpt]