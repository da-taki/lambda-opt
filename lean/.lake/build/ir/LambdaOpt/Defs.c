// Lean compiler output
// Module: LambdaOpt.Defs
// Imports: public import Init public import Mathlib.Topology.MetricSpace.Basic public import Mathlib.Topology.MetricSpace.Lipschitz
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_trajectory___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_trajectory___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_trajectory(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_trajectory___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_divergence___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_divergence___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_divergence(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_divergence___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_trajectory___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_nat_dec_eq(x_3, x_4);
if (x_5 == 1)
{
lean_dec(x_1);
lean_inc(x_2);
return x_2;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_unsigned_to_nat(1u);
x_7 = lean_nat_sub(x_3, x_6);
lean_inc(x_1);
x_8 = lp_lambda_x2dopt_LambdaOpt_trajectory___redArg(x_1, x_2, x_7);
lean_dec(x_7);
x_9 = lean_apply_1(x_1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_trajectory___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_lambda_x2dopt_LambdaOpt_trajectory___redArg(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_trajectory(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_lambda_x2dopt_LambdaOpt_trajectory___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_trajectory___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_lambda_x2dopt_LambdaOpt_trajectory(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = lean_nat_dec_lt(x_5, x_4);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc(x_1);
x_7 = lp_lambda_x2dopt_LambdaOpt_trajectory___redArg(x_1, x_3, x_4);
x_8 = lean_apply_1(x_2, x_7);
x_9 = lean_nat_sub(x_5, x_4);
x_10 = lp_lambda_x2dopt_LambdaOpt_trajectory___redArg(x_1, x_8, x_9);
lean_dec(x_9);
lean_dec(x_8);
return x_10;
}
else
{
lean_object* x_11; 
lean_dec(x_2);
x_11 = lp_lambda_x2dopt_LambdaOpt_trajectory___redArg(x_1, x_3, x_5);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_divergence___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec_ref(x_1);
lean_inc(x_2);
x_8 = lp_lambda_x2dopt_LambdaOpt_rewrittenTrajectory___redArg(x_2, x_3, x_4, x_5, x_6);
x_9 = lp_lambda_x2dopt_LambdaOpt_trajectory___redArg(x_2, x_4, x_6);
x_10 = lean_apply_2(x_7, x_8, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_divergence___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_lambda_x2dopt_LambdaOpt_divergence___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_divergence(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_lambda_x2dopt_LambdaOpt_divergence___redArg(x_2, x_3, x_4, x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_divergence___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_lambda_x2dopt_LambdaOpt_divergence(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_MetricSpace_Lipschitz(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_lambda_x2dopt_LambdaOpt_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_MetricSpace_Lipschitz(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
