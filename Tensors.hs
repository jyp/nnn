{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds, TypeFamilies, PolyKinds, ScopedTypeVariables #-}

import GHC.TypeLits
import Data.Kind

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

data Tensor (p :: Nat) -- dimensions of parameters tensor ("free variables")
            (d :: Nat) -- dimensions of result.
  = Tensor {value :: Pull ('Dim 1 d) -- value
           ,grad :: Pull ('Dim p d)  -- gradient wrt. the parameters p
           }

data Sing :: a -> Type

idT :: Pull ('Dim d d)
idT index pointer = error "if i = j then 1 else 0"


parameters :: Pull ('Dim 1 p) -> Tensor p p -- input parameters in the computation.
parameters x = Tensor x idT


(·) :: Pull ('Dim d e) -> Pull ('Dim e f) -> Pull ('Dim d f)
(·) = error "matrix product"


-- VERY IMPORTANT NOTE: the parameters of the input (y) can't be
-- passed implicitly ("in a closure environment") to the body of "f".
-- Fortunately the typesystem will prevent this (because of the inner
-- quantification of "q").

-- This is not too annoying in a DL context, but we'd really want the
-- function "f" to introduce its own parameters. (The weights for the
-- neurons of the layers embedded there.) For simplicity, we may want to have another
-- set of parameters introduced at the same 
chain :: forall p d e. Tensor p d -> (forall q. Tensor q d -> Tensor q e) -> Tensor p e
chain (Tensor y dy) f = Tensor z (dy · dz)
  where (Tensor z dz) = f (Tensor y idT)
        z :: Pull ('Dim 1 e) -- final values
        dz :: Pull ('Dim d e) -- the gradient of z wrt. the result of y
        -- y :: Pull ('Dim 1 d) -- intermediate values
        -- dy :: Pull ('Dim p d) -- gradient of y wrt. the original params ("intermediate gradient")
