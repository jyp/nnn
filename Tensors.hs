{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds, TypeFamilies, PolyKinds, ScopedTypeVariables #-}

import GHC.TypeLits
import Data.Kind
import Algebra.Classes
import Prelude hiding (Num(..))

data Pointer -- location in (GPU) memory. We assume that all values are (say) Floats for now.
  -- Todo: Add a parameter everywhere.
data Index d -- representation of an index in a tensor of d dimensions
data Code -- representation of code
type N a = a -> Code
type NN a = N (N a)  -- batman!
type Value = N Pointer -- "consuming a pointer" ≅ writing in it
type Pull d = Index d -> Value -- how to read the given index into the given pointer.
  -- Will probably need to be doubly negated to allow for allocation of intermediate arrays.
type Push d = N (Pull d)

data Dim = Dim Nat Nat -- inner and outer dimensions

data Grads (d :: Nat) (ps :: [Nat]) where
  Nil :: Grads d '[]
  (:>) :: Pull ('Dim p d) -> Grads d ps -> Grads d (p ': ps)

instance Additive (Grads d ps) where

mulGrads :: Grads e ps -> e×d -> Grads d ps
mulGrads Nil _ = Nil
mulGrads (g :> gs) m = g · m :> mulGrads gs m

data Tensor (ps :: [Nat]) -- dimensions of parameter tensors ("free variables")
            (d :: Nat) -- dimensions of result.
  = Tensor {value :: Pull ('Dim 1 d) -- value
           ,grad :: Grads d ps  -- gradient wrt. the parameters ps
           }

idT :: Pull ('Dim d d)
idT index pointer = error "if i = j then 1 else 0"

var :: Pull ('Dim 1 p) -> Tensor '[p] p -- input parameters in the computation.
var x = Tensor x (idT :> Nil)


(·) :: Pull ('Dim d e) -> Pull ('Dim e f) -> Pull ('Dim d f)
(·) = error "matrix product"


type a × b = Pull ('Dim a b)

-- chain' :: forall p d e r. Tensor '[p] d -> (forall q. Tensor '[q]  d -> Tensor (q ': r) e) -> Tensor (p ': r) e
-- chain' (Tensor (y :: 1×d) ((dydp :: p×d) :> Nil)) f = case f (Tensor y (idT :> Nil)) of
--   Tensor (z :: 1×e) (dpdy :> dzdr) -> Tensor z (dydp · dpdy :> dzdr)

chain :: forall d e r. Tensor r d -> (forall q. Tensor (q : r)  d -> Tensor (q : r) e) -> Tensor r e
chain (Tensor (y :: 1×d) dydr) f = case f (Tensor y (idT :> dydr)) of
  Tensor (z :: 1×e) (dpdy :> dzdr) -> Tensor z (mulGrads dydr dpdy + dzdr)

-- param1 :: 1×p -> Tensor [q] e -> Tensor [q,p] e
-- param1 v t = 

-- data Expr ps d where
--   Bind :: Expr ps d -> (Expr ps d -> Expr ps e) -> Expr ps e
--   Param0 :: Expr (d:ps) d
--   Param1 :: Expr (p0:d:ps) d

