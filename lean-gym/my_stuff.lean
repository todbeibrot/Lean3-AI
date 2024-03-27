import tactic

-- Replacements for inbuild types. This way we can use `apply yyy_to`. `apply →` doesn't work
@[reducible] def yyy_to (a b : Prop) := a → b
@[reducible] def zzz_forall  {α : Sort*} (a : α → Prop) := ∀(x : α), a x
@[reducible] def www_fun (α : Sort*) {β : Sort*} {f : α → β} := λ (a : α), f a

@[simp] lemma yyy_to_def (a b : Prop) : yyy_to a b = (a → b) := rfl
@[simp] lemma zzz_forall_def {α : Sort*} (a : α → Prop) : zzz_forall a = ∀(x : α), a x := rfl
@[simp] lemma www_fun_def (α : Sort*) {β : Sort*} {f : α → β} : @www_fun α β f = λ (a : α), f a := rfl

-- `aaa` protect the goals till we want to work on them
lemma aaa (p : Prop) : Prop := p
def ccc_goal (x : Sort*) := x
lemma aaa_unfold (p : Prop) : aaa p = p := sorry
lemma aaa_unfold_neg (p : Prop) : aaa p = ¬p := sorry

structure my_struc :=
(l : Prop)
(p1 : aaa l)


namespace tactic
namespace interactive

setup_tactic_parser

open suggest
open solve_by_elim

meta def get_uninitialized_vars_aux : list expr → list expr → tactic (list expr)
| [] l := return l
| (a :: l1) l2 := (do t ← infer_type a,
                 tactic.unify t `(true),
                 get_uninitialized_vars_aux l1 (a :: l2) ) <|> get_uninitialized_vars_aux l1 l2

meta def get_uninitialized_vars : tactic (list expr) :=
do ctx ← local_context,
   r ← capture skip,
   uninitialized_vars ← get_uninitialized_vars_aux ctx [],
   resume r,
   return uninitialized_vars

meta def list_inter : list expr → list expr → tactic unit
| [] [] := skip
| l1 [] := skip
| [] l2 := skip
| (a :: l1) (b :: l2) := do tactic.fail_if_success (tactic.unify a b),
                            list_inter (a :: l1) l2,
                            list_inter l1 (b :: l2)

meta def lbs (l : list expr) (opt : suggest_opt := { }) : tactic string :=
do
(suggest_core opt).mfirst (λ a, do
  guard (a.num_goals = 0),
  write a.state,
  list_inter l a.hyps_used,
  return a.script)

-- like library_search but I don't want to exclude uninitialized
meta def my_library_search (semireducible : parse $ optional (tk "!"))
  (hs : parse simp_arg_list) (attr_names : parse with_ident_list)
  (use : parse $ (tk "using" *> many ident_) <|> return [])
  (opt : suggest.suggest_opt := { }) : tactic unit :=
do l ← get_uninitialized_vars,
   (lemma_thunks, ctx_thunk) ← mk_assumption_set ff hs attr_names,
   use ← use.mmap get_local,
   (lbs l
     { compulsory_hyps := use,
       backtrack_all_goals := tt,
       lemma_thunks := some lemma_thunks,
       ctx_thunk := ctx_thunk,
       md := if semireducible.is_some then
         tactic.transparency.semireducible else tactic.transparency.reducible,
       ..opt } >>=
   (λ _, skip)) <|>
   tactic.fail "library_search failed"


-- like assumption but we reverse the order of the local variables
meta def my_assumption : tactic unit :=
do { ctx ← local_context,
     t   ← target,
     H   ← find_same_type t (list.reverse ctx),
     tactic.exact H }
<|> fail "assumption tactic failed"


meta def close_goal_aux : list expr → tactic unit
| [] := assumption
| (a :: l) := do assumption <|> do tactic.revert a, close_goal_aux l

-- tries to close the main goal by using local variables
meta def close_goal : tactic unit :=
do ctx ← local_context,
   close_goal_aux (list.reverse ctx)


meta def clear_trivial_aux : list expr → tactic unit
| [] := skip
| (a :: l) := do t ← infer_type a,
                 tactic.try (do tactic.unify t `(true),
                                tactic.clear a),
                 clear_trivial_aux l

-- clears all local variables of type `true`
meta def clear_trivial : tactic unit :=
do ctx ← local_context,
   clear_trivial_aux ctx


open tactic.ring

-- like ring but without the `Try this: ...` message
meta def my_ring (red : interactive.parse (lean.parser.tk "!")?) : tactic unit :=
ring1 red <|>
(ring_nf red normalize_mode.horner (loc.ns [none]))

end interactive
end tactic
