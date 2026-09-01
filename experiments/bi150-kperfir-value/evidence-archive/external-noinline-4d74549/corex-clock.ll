; ModuleID = '/tmp/kernelswift-route-c-4d74549e6747e289ee80989bcdd7fe45770ef07c/experiments/bi150-kperfir-value/device/corex_clock.cu'
source_filename = "/tmp/kernelswift-route-c-4d74549e6747e289ee80989bcdd7fe45770ef07c/experiments/bi150-kperfir-value/device/corex_clock.cu"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
target triple = "bi-iluvatar-ilurt"

@gSyncThreadsFlag = dso_local addrspace(3) global i32 undef, align 4
@llvm.compiler.used = appending global [6 x ptr] [ptr @corex_clock64_start, ptr @corex_clock64_after_u64, ptr addrspacecast (ptr addrspace(1) @gAssertReason to ptr), ptr addrspacecast (ptr addrspace(1) @gMallocHeapBase to ptr), ptr addrspacecast (ptr addrspace(1) @gMallocHeapSize to ptr), ptr addrspacecast (ptr addrspace(1) @gPrintBuf to ptr)], section "llvm.metadata"
@gPrintBuf = weak dso_local addrspace(1) externally_initialized global ptr null, align 8
@gAssertReason = weak dso_local addrspace(1) externally_initialized global ptr null, align 8
@gMallocHeapBase = weak addrspace(1) externally_initialized global ptr null, align 8
@gMallocHeapSize = weak addrspace(1) externally_initialized global i32 0, align 4

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local i64 @corex_clock64_start() #0 {
  %1 = alloca i64, align 8, addrspace(5)
  %2 = alloca i64, align 8, addrspace(5)
  %3 = addrspacecast ptr addrspace(5) %2 to ptr
  %4 = addrspacecast ptr addrspace(5) %1 to ptr
  %5 = call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()
  ret i64 %5
}

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local i64 @corex_clock64_after_u64(i64 noundef %0) #0 {
  %2 = alloca i64, align 8, addrspace(5)
  %3 = alloca i64, align 8, addrspace(5)
  %4 = alloca i64, align 8, addrspace(5)
  %5 = addrspacecast ptr addrspace(5) %3 to ptr
  %6 = addrspacecast ptr addrspace(5) %4 to ptr
  store i64 %0, ptr %6, align 8
  %7 = addrspacecast ptr addrspace(5) %2 to ptr
  %8 = call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()
  %9 = load i64, ptr %6, align 8
  %10 = and i64 %9, 1
  %11 = add i64 %8, %10
  ret i64 %11
}

; Function Attrs: nocallback nounwind memory(inaccessiblemem: readwrite)
declare noundef i64 @llvm.nvvm.read.ptx.sreg.clock64() #1

attributes #0 = { convergent mustprogress noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="ivcore11" "target-features"="+async-copy-sme,+mma-i8-mn16k32,+pipe-bar,+v4i8-alu" }
attributes #1 = { nocallback nounwind memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 18.1.8 (4.4.0 862f87fe94c5eaa2928bda965ed6abb85a25eb7c)"}
