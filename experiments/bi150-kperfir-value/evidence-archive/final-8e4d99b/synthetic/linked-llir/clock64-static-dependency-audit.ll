; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "bi-iluvatar-ilurt"

@gPrintBuf = weak dso_local addrspace(1) externally_initialized global ptr null, align 8
@gAssertReason = weak dso_local addrspace(1) externally_initialized global ptr null, align 8
@gMallocHeapBase = weak addrspace(1) externally_initialized global ptr null, align 8
@gMallocHeapSize = weak addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [6 x ptr] [ptr @corex_clock64_after_u64, ptr @corex_clock64_start, ptr addrspacecast (ptr addrspace(1) @gAssertReason to ptr), ptr addrspacecast (ptr addrspace(1) @gMallocHeapBase to ptr), ptr addrspacecast (ptr addrspace(1) @gMallocHeapSize to ptr), ptr addrspacecast (ptr addrspace(1) @gPrintBuf to ptr)], section "llvm.metadata"

; Function Attrs: nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define iluvatar_kernel void @external_clock_kernel(ptr addrspace(1) nocapture readonly %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %3) local_unnamed_addr #0 !dbg !10 {
  %5 = addrspacecast ptr addrspace(1) %0 to ptr, !dbg !13
  %6 = load <1 x i64>, ptr %5, align 8, !dbg !13
  %7 = tail call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64(), !dbg !14
  %8 = tail call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64(), !dbg !17
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !19
  %10 = icmp eq i32 %9, 0, !dbg !19
  br i1 %10, label %.critedge, label %.critedge2, !dbg !19

.critedge:                                        ; preds = %4
  %11 = extractelement <1 x i64> %6, i64 0, !dbg !13
  %12 = shl i64 %11, 13, !dbg !20
  %13 = xor i64 %12, %11, !dbg !21
  %14 = add i64 %13, 25214903917, !dbg !22
  %15 = lshr i64 %14, 7, !dbg !23
  %16 = xor i64 %15, %14, !dbg !24
  %17 = shl i64 %16, 13, !dbg !20
  %18 = xor i64 %17, %16, !dbg !21
  %19 = add i64 %18, 25214903917, !dbg !22
  %20 = lshr i64 %19, 7, !dbg !23
  %21 = xor i64 %20, %19, !dbg !24
  %22 = shl i64 %21, 13, !dbg !20
  %23 = xor i64 %22, %21, !dbg !21
  %24 = add i64 %23, 25214903917, !dbg !22
  %25 = lshr i64 %24, 7, !dbg !23
  %26 = xor i64 %25, %24, !dbg !24
  %27 = shl i64 %26, 13, !dbg !20
  %28 = xor i64 %27, %26, !dbg !21
  %29 = add i64 %28, 25214903917, !dbg !22
  %30 = lshr i64 %29, 7, !dbg !23
  %31 = xor i64 %30, %29, !dbg !24
  %32 = shl i64 %31, 13, !dbg !20
  %33 = xor i64 %32, %31, !dbg !21
  %34 = add i64 %33, 25214903917, !dbg !22
  %35 = lshr i64 %34, 7, !dbg !23
  %36 = xor i64 %35, %34, !dbg !24
  %37 = shl i64 %36, 13, !dbg !20
  %38 = xor i64 %37, %36, !dbg !21
  %39 = add i64 %38, 25214903917, !dbg !22
  %40 = lshr i64 %39, 7, !dbg !23
  %41 = xor i64 %40, %39, !dbg !24
  %42 = shl i64 %41, 13, !dbg !20
  %43 = xor i64 %42, %41, !dbg !21
  %44 = add i64 %43, 25214903917, !dbg !22
  %45 = lshr i64 %44, 7, !dbg !23
  %46 = xor i64 %45, %44, !dbg !24
  %47 = shl i64 %46, 13, !dbg !20
  %48 = xor i64 %47, %46, !dbg !21
  %49 = add i64 %48, 25214903917, !dbg !22
  %50 = lshr i64 %49, 7, !dbg !23
  %51 = xor i64 %50, %49, !dbg !24
  %52 = shl i64 %51, 13, !dbg !20
  %53 = xor i64 %52, %51, !dbg !21
  %54 = add i64 %53, 25214903917, !dbg !22
  %55 = lshr i64 %54, 7, !dbg !23
  %56 = xor i64 %55, %54, !dbg !24
  %57 = shl i64 %56, 13, !dbg !20
  %58 = xor i64 %57, %56, !dbg !21
  %59 = add i64 %58, 25214903917, !dbg !22
  %60 = lshr i64 %59, 7, !dbg !23
  %61 = xor i64 %60, %59, !dbg !24
  %62 = shl i64 %61, 13, !dbg !20
  %63 = xor i64 %62, %61, !dbg !21
  %64 = add i64 %63, 25214903917, !dbg !22
  %65 = lshr i64 %64, 7, !dbg !23
  %66 = xor i64 %65, %64, !dbg !24
  %67 = shl i64 %66, 13, !dbg !20
  %68 = xor i64 %67, %66, !dbg !21
  %69 = add i64 %68, 25214903917, !dbg !22
  %70 = lshr i64 %69, 7, !dbg !23
  %71 = xor i64 %70, %69, !dbg !24
  %72 = shl i64 %71, 13, !dbg !20
  %73 = xor i64 %72, %71, !dbg !21
  %74 = add i64 %73, 25214903917, !dbg !22
  %75 = lshr i64 %74, 7, !dbg !23
  %76 = xor i64 %75, %74, !dbg !24
  %77 = shl i64 %76, 13, !dbg !20
  %78 = xor i64 %77, %76, !dbg !21
  %79 = add i64 %78, 25214903917, !dbg !22
  %80 = lshr i64 %79, 7, !dbg !23
  %81 = xor i64 %80, %79, !dbg !24
  %82 = shl i64 %81, 13, !dbg !20
  %83 = xor i64 %82, %81, !dbg !21
  %84 = add i64 %83, 25214903917, !dbg !22
  %85 = lshr i64 %84, 7, !dbg !23
  %86 = xor i64 %85, %84, !dbg !24
  %87 = shl i64 %86, 13, !dbg !20
  %88 = xor i64 %87, %86, !dbg !21
  %89 = add i64 %88, 25214903917, !dbg !22
  %90 = lshr i64 %89, 7, !dbg !23
  %91 = xor i64 %90, %89, !dbg !24
  tail call void @llvm.bi.store.kop.i64(i64 %91, ptr addrspace(1) %2, i32 0, i1 false), !dbg !19
  %92 = sext i32 %3 to i64, !dbg !25
  tail call void @llvm.bi.store.kop.i64(i64 %92, ptr addrspace(1) %1, i32 0, i1 false), !dbg !25
  %93 = getelementptr i64, ptr addrspace(1) %1, i64 1, !dbg !26
  tail call void @llvm.bi.store.kop.i64(i64 %7, ptr addrspace(1) %93, i32 0, i1 false), !dbg !27
  %94 = getelementptr i64, ptr addrspace(1) %1, i64 2, !dbg !28
  tail call void @llvm.bi.store.kop.i64(i64 %8, ptr addrspace(1) %94, i32 0, i1 false), !dbg !29
  br label %.critedge2, !dbg !29

.critedge2:                                       ; preds = %4, %.critedge
  ret void, !dbg !30
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nounwind memory(argmem: write)
declare void @llvm.bi.store.kop.i64(i64, ptr addrspace(1) writeonly, i32 immarg, i1 immarg) #2

; Function Attrs: alwaysinline mustprogress norecurse nounwind memory(inaccessiblemem: readwrite)
define internal noundef i64 @corex_clock64_after_u64(i64 noundef %0) #3 {
  %2 = and i64 %0, 1
  %3 = tail call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()
  %4 = add i64 %3, %2
  ret i64 %4
}

; Function Attrs: alwaysinline mustprogress norecurse nounwind memory(inaccessiblemem: readwrite)
define internal noundef i64 @corex_clock64_start() #3 {
  %1 = tail call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()
  ret i64 %1
}

; Function Attrs: nocallback nounwind memory(inaccessiblemem: readwrite)
declare noundef i64 @llvm.nvvm.read.ptx.sreg.clock64() #4

attributes #0 = { nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nounwind memory(argmem: write) }
attributes #3 = { alwaysinline mustprogress norecurse nounwind memory(inaccessiblemem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="ivcore11" "target-features"="+async-copy-sme,+mma-i8-mn16k32,+pipe-bar,+v4i8-alu" }
attributes #4 = { nocallback nounwind memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.dbg.cu = !{!5}
!nvvm.annotations = !{!7, !8}
!llvm.ident = !{!9}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 2]}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!6 = !DIFile(filename: "clock64_probe.py", directory: "/tmp/kernelswift-route-c-8e4d99b89407ecb3d35bac1c276d3cc73de27699/experiments/bi150-kperfir-value/scripts")
!7 = !{ptr @external_clock_kernel, !"kernel", i32 1}
!8 = !{ptr @external_clock_kernel, !"maxntidx", i32 32}
!9 = !{!"clang version 18.1.8 (4.4.0 862f87fe94c5eaa2928bda965ed6abb85a25eb7c)"}
!10 = distinct !DISubprogram(name: "external_clock_kernel", linkageName: "external_clock_kernel", scope: !6, file: !6, line: 319, type: !11, scopeLine: 319, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5)
!11 = !DISubroutineType(cc: DW_CC_normal, types: !12)
!12 = !{}
!13 = !DILocation(line: 326, column: 18, scope: !10)
!14 = !DILocation(line: 303, column: 8, scope: !15, inlinedAt: !16)
!15 = distinct !DILexicalBlockFile(scope: !10, file: !6, discriminator: 0)
!16 = !DILocation(line: 328, column: 18, scope: !10)
!17 = !DILocation(line: 313, column: 8, scope: !15, inlinedAt: !18)
!18 = !DILocation(line: 334, column: 33, scope: !10)
!19 = !DILocation(line: 336, column: 25, scope: !10)
!20 = !DILocation(line: 330, column: 28, scope: !10)
!21 = !DILocation(line: 330, column: 21, scope: !10)
!22 = !DILocation(line: 331, column: 20, scope: !10)
!23 = !DILocation(line: 332, column: 28, scope: !10)
!24 = !DILocation(line: 332, column: 21, scope: !10)
!25 = !DILocation(line: 337, column: 26, scope: !10)
!26 = !DILocation(line: 338, column: 27, scope: !10)
!27 = !DILocation(line: 338, column: 30, scope: !10)
!28 = !DILocation(line: 339, column: 27, scope: !10)
!29 = !DILocation(line: 339, column: 30, scope: !10)
!30 = !DILocation(line: 339, column: 4, scope: !10)
