// Lean compiler output
// Module: LambdaOpt.Decidable
// Imports: public import Init public import Mathlib.Data.Real.Basic public import Mathlib.Tactic
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
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_computeBound_spec__1_spec__1(lean_object*, lean_object*);
lean_object* lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_1138242547____hygCtx___hyg_8_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_computeBound_spec__1_spec__1___boxed(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Real_definition_00___x40_Mathlib_Data_Real_Basic_1850581184____hygCtx___hyg_8_;
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_sum___at___00LambdaOpt_computeBound_spec__1(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_mathlib_npowRec___at___00Cardinal_cantorFunctionAux_spec__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_4214226450____hygCtx___hyg_8_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_mapTR_loop___at___00LambdaOpt_computeBound_spec__0(lean_object*, lean_object*, lean_object*);
lean_object* l_List_zipWith___at___00List_zip_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_computeBound(lean_object*, lean_object*, lean_object*);
lean_object* l_Rat_pow(lean_object*, lean_object*);
lean_object* l_Rat_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_mapTR_loop___at___00LambdaOpt_rationalBoundDecidable_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1_spec__1(lean_object*, lean_object*);
lean_object* l_Rat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1_spec__1___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_cast___at___00Tactic_NormNum_evalRealSqrt_spec__3(lean_object*);
static lean_once_cell_t lp_lambda_x2dopt_List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_lambda_x2dopt_List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1___closed__0;
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1(lean_object*);
uint8_t l_Rat_instDecidableLe(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_lambda_x2dopt_LambdaOpt_rationalBoundDecidable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_rationalBoundDecidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_computeBound_spec__1_spec__1(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_inc(x_1);
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_computeBound_spec__1_spec__1(x_1, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_1138242547____hygCtx___hyg_8_), 3, 2);
lean_closure_set(x_6, 0, x_3);
lean_closure_set(x_6, 1, x_5);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_computeBound_spec__1_spec__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_computeBound_spec__1_spec__1(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_sum___at___00LambdaOpt_computeBound_spec__1(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Real_definition_00___x40_Mathlib_Data_Real_Basic_1850581184____hygCtx___hyg_8_;
x_3 = lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_computeBound_spec__1_spec__1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_mapTR_loop___at___00LambdaOpt_computeBound_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec(x_1);
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 1);
lean_inc(x_9);
lean_dec(x_6);
lean_inc(x_1);
x_10 = lp_mathlib_npowRec___at___00Cardinal_cantorFunctionAux_spec__0(x_9, x_1);
lean_dec(x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_4214226450____hygCtx___hyg_8_), 3, 2);
lean_closure_set(x_11, 0, x_8);
lean_closure_set(x_11, 1, x_10);
lean_ctor_set(x_2, 1, x_3);
lean_ctor_set(x_2, 0, x_11);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_13 = lean_ctor_get(x_2, 0);
x_14 = lean_ctor_get(x_2, 1);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_2);
x_15 = lean_ctor_get(x_13, 0);
lean_inc(x_15);
x_16 = lean_ctor_get(x_13, 1);
lean_inc(x_16);
lean_dec(x_13);
lean_inc(x_1);
x_17 = lp_mathlib_npowRec___at___00Cardinal_cantorFunctionAux_spec__0(x_16, x_1);
lean_dec(x_16);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Real_definition___lam__0_00___x40_Mathlib_Data_Real_Basic_4214226450____hygCtx___hyg_8_), 3, 2);
lean_closure_set(x_18, 0, x_15);
lean_closure_set(x_18, 1, x_17);
x_19 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_3);
x_2 = x_14;
x_3 = x_19;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_computeBound(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = l_List_zipWith___at___00List_zip_spec__0___redArg(x_2, x_3);
x_5 = lean_box(0);
x_6 = lp_lambda_x2dopt_List_mapTR_loop___at___00LambdaOpt_computeBound_spec__0(x_1, x_4, x_5);
x_7 = lp_lambda_x2dopt_List_sum___at___00LambdaOpt_computeBound_spec__1(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_mapTR_loop___at___00LambdaOpt_rationalBoundDecidable_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec_ref(x_1);
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 1);
lean_inc(x_9);
lean_dec(x_6);
lean_inc_ref(x_1);
x_10 = l_Rat_pow(x_1, x_9);
lean_dec(x_9);
x_11 = l_Rat_mul(x_8, x_10);
lean_dec(x_8);
lean_ctor_set(x_2, 1, x_3);
lean_ctor_set(x_2, 0, x_11);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_13 = lean_ctor_get(x_2, 0);
x_14 = lean_ctor_get(x_2, 1);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_2);
x_15 = lean_ctor_get(x_13, 0);
lean_inc(x_15);
x_16 = lean_ctor_get(x_13, 1);
lean_inc(x_16);
lean_dec(x_13);
lean_inc_ref(x_1);
x_17 = l_Rat_pow(x_1, x_16);
lean_dec(x_16);
x_18 = l_Rat_mul(x_15, x_17);
lean_dec(x_15);
x_19 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_3);
x_2 = x_14;
x_3 = x_19;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1_spec__1(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_inc_ref(x_1);
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1_spec__1(x_1, x_4);
x_6 = l_Rat_add(x_3, x_5);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1_spec__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1_spec__1(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
static lean_object* _init_lp_lambda_x2dopt_List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1___closed__0(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lp_mathlib_Nat_cast___at___00Tactic_NormNum_evalRealSqrt_spec__3(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_obj_once(&lp_lambda_x2dopt_List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1___closed__0, &lp_lambda_x2dopt_List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1___closed__0_once, _init_lp_lambda_x2dopt_List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1___closed__0);
x_3 = lp_lambda_x2dopt_List_foldr___at___00List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1_spec__1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_lambda_x2dopt_LambdaOpt_rationalBoundDecidable(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_5 = l_List_zipWith___at___00List_zip_spec__0___redArg(x_2, x_3);
x_6 = lean_box(0);
x_7 = lp_lambda_x2dopt_List_mapTR_loop___at___00LambdaOpt_rationalBoundDecidable_spec__0(x_1, x_5, x_6);
x_8 = lp_lambda_x2dopt_List_sum___at___00LambdaOpt_rationalBoundDecidable_spec__1(x_7);
x_9 = l_Rat_instDecidableLe(x_8, x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_lambda_x2dopt_LambdaOpt_rationalBoundDecidable___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_lambda_x2dopt_LambdaOpt_rationalBoundDecidable(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Real_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_lambda_x2dopt_LambdaOpt_Decidable(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Real_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
