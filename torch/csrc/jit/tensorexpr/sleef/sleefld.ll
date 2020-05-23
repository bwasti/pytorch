; ModuleID = '/home/bwasti/pytorch/sleef/src/libm/sleefld.c'
source_filename = "/home/bwasti/pytorch/sleef/src/libm/sleefld.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.Sleef_longdouble2 = type { x86_fp80, x86_fp80 }

; Function Attrs: nounwind uwtable
define void @Sleef_sincospil_u05(%struct.Sleef_longdouble2* noalias nocapture sret, x86_fp80) local_unnamed_addr #0 {
  %3 = alloca [6 x i8], align 2
  %4 = alloca [6 x i8], align 2
  %5 = getelementptr inbounds [6 x i8], [6 x i8]* %3, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 6, i8* nonnull %5)
  %6 = getelementptr inbounds [6 x i8], [6 x i8]* %4, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 6, i8* nonnull %6)
  %7 = fmul x86_fp80 %1, 0xK40018000000000000000
  %8 = fptosi x86_fp80 %7 to i64
  %9 = fcmp uge x86_fp80 %7, 0xK00000000000000000000
  %10 = zext i1 %9 to i64
  %11 = add nsw i64 %10, %8
  %12 = and i64 %11, -2
  %13 = sitofp i64 %12 to x86_fp80
  %14 = fsub x86_fp80 %7, %13
  %15 = fmul x86_fp80 %14, %14
  %16 = bitcast x86_fp80 %14 to i80
  %17 = and i80 %16, -4294967296
  %18 = bitcast i80 %17 to x86_fp80
  %19 = fsub x86_fp80 %14, %18
  %20 = fmul x86_fp80 %18, %18
  %21 = fsub x86_fp80 %20, %15
  %22 = fmul x86_fp80 %19, %18
  %23 = fadd x86_fp80 %22, %21
  %24 = fadd x86_fp80 %22, %23
  %25 = fmul x86_fp80 %19, %19
  %26 = fadd x86_fp80 %25, %24
  %27 = fmul x86_fp80 %15, 0xK3FC8D3CC8344C0375BBB
  %28 = fadd x86_fp80 %27, 0xKBFD1B7D55DAA7CC12DB6
  %29 = fmul x86_fp80 %15, %28
  %30 = fadd x86_fp80 %29, 0xK3FD9F47A18F3C23A5579
  %31 = fmul x86_fp80 %15, %30
  %32 = fadd x86_fp80 %31, 0xKBFE1F183A7EE716C340F
  %33 = fmul x86_fp80 %15, %32
  %34 = fadd x86_fp80 %33, 0xK3FE9A83C1A43F6F5D82D
  %35 = fmul x86_fp80 %15, %34
  %36 = fadd x86_fp80 %35, 0xKBFF09969667315EC1395
  %37 = fmul x86_fp80 %15, %36
  %38 = fadd x86_fp80 %37, 0xK3FF6A335E33BAD570E88
  %39 = fmul x86_fp80 %15, %38
  %40 = fadd x86_fp80 %39, 0xKBFFBA55DE7312DF295F5
  %41 = fsub x86_fp80 %40, %39
  %42 = fsub x86_fp80 %40, %41
  %43 = fsub x86_fp80 %39, %42
  %44 = fsub x86_fp80 0xKBFFBA55DE7312DF295F5, %41
  %45 = fadd x86_fp80 %44, %43
  %46 = fadd x86_fp80 %45, 0xKBFBAB5796327B4720000
  %47 = bitcast x86_fp80 %15 to i80
  %48 = and i80 %47, -4294967296
  %49 = bitcast i80 %48 to x86_fp80
  %50 = fsub x86_fp80 %15, %49
  %51 = bitcast x86_fp80 %40 to i80
  %52 = and i80 %51, -4294967296
  %53 = bitcast i80 %52 to x86_fp80
  %54 = fsub x86_fp80 %40, %53
  %55 = fmul x86_fp80 %15, %40
  %56 = fmul x86_fp80 %49, %53
  %57 = fsub x86_fp80 %56, %55
  %58 = fmul x86_fp80 %50, %53
  %59 = fadd x86_fp80 %58, %57
  %60 = fmul x86_fp80 %54, %49
  %61 = fadd x86_fp80 %60, %59
  %62 = fmul x86_fp80 %50, %54
  %63 = fadd x86_fp80 %62, %61
  %64 = fmul x86_fp80 %15, %46
  %65 = fadd x86_fp80 %64, %63
  %66 = fmul x86_fp80 %26, %40
  %67 = fadd x86_fp80 %66, %65
  %68 = fadd x86_fp80 %55, 0xK3FFEC90FDAA22168C235
  %69 = fsub x86_fp80 %68, %55
  %70 = fsub x86_fp80 %68, %69
  %71 = fsub x86_fp80 %55, %70
  %72 = fsub x86_fp80 0xK3FFEC90FDAA22168C235, %69
  %73 = fadd x86_fp80 %72, %71
  %74 = fadd x86_fp80 %67, 0xKBFBCECE989D49A080000
  %75 = fadd x86_fp80 %73, %74
  %76 = bitcast x86_fp80 %68 to i80
  %77 = and i80 %76, -4294967296
  %78 = bitcast i80 %77 to x86_fp80
  %79 = fsub x86_fp80 %68, %78
  %80 = fmul x86_fp80 %14, %68
  %81 = fmul x86_fp80 %18, %78
  %82 = fsub x86_fp80 %81, %80
  %83 = fmul x86_fp80 %79, %18
  %84 = fadd x86_fp80 %83, %82
  %85 = fmul x86_fp80 %19, %78
  %86 = fadd x86_fp80 %85, %84
  %87 = fmul x86_fp80 %19, %79
  %88 = fadd x86_fp80 %87, %86
  %89 = fmul x86_fp80 %14, %75
  %90 = fadd x86_fp80 %88, %89
  %91 = fadd x86_fp80 %80, %90
  %92 = fmul x86_fp80 %15, 0xK3FC493E30439D478F07F
  %93 = fsub x86_fp80 0xK3FCD9061F3FA4C54B6AE, %92
  %94 = fmul x86_fp80 %15, %93
  %95 = fadd x86_fp80 %94, 0xKBFD5DB71266409051953
  %96 = fmul x86_fp80 %15, %95
  %97 = fadd x86_fp80 %96, 0xK3FDDFCE9C51AE0707625
  %98 = fmul x86_fp80 %15, %97
  %99 = fadd x86_fp80 %98, 0xKBFE5D368F95101FFEF42
  %100 = fmul x86_fp80 %15, %99
  %101 = fadd x86_fp80 %100, 0xK3FECF0FA83448DD5AED0
  %102 = fmul x86_fp80 %15, %101
  %103 = fadd x86_fp80 %102, 0xKBFF3AAE9E3F1E5FFCFD8
  %104 = fmul x86_fp80 %15, %103
  %105 = fadd x86_fp80 %104, 0xK3FF981E0F840DAD61D9B
  %106 = fsub x86_fp80 %105, %104
  %107 = fsub x86_fp80 %105, %106
  %108 = fsub x86_fp80 %104, %107
  %109 = fsub x86_fp80 0xK3FF981E0F840DAD61D9B, %106
  %110 = fadd x86_fp80 %109, %108
  %111 = fadd x86_fp80 %110, 0xKBFB8D2D2CADD8FAE0000
  %112 = bitcast x86_fp80 %105 to i80
  %113 = and i80 %112, -4294967296
  %114 = bitcast i80 %113 to x86_fp80
  %115 = fsub x86_fp80 %105, %114
  %116 = fmul x86_fp80 %15, %105
  %117 = fmul x86_fp80 %49, %114
  %118 = fsub x86_fp80 %117, %116
  %119 = fmul x86_fp80 %50, %114
  %120 = fadd x86_fp80 %119, %118
  %121 = fmul x86_fp80 %115, %49
  %122 = fadd x86_fp80 %121, %120
  %123 = fmul x86_fp80 %50, %115
  %124 = fadd x86_fp80 %123, %122
  %125 = fmul x86_fp80 %15, %111
  %126 = fadd x86_fp80 %125, %124
  %127 = fmul x86_fp80 %26, %105
  %128 = fadd x86_fp80 %127, %126
  %129 = fadd x86_fp80 %116, 0xKBFFD9DE9E64DF22EF2D2
  %130 = fsub x86_fp80 %129, %116
  %131 = fsub x86_fp80 %129, %130
  %132 = fsub x86_fp80 %116, %131
  %133 = fsub x86_fp80 0xKBFFD9DE9E64DF22EF2D2, %130
  %134 = fadd x86_fp80 %133, %132
  %135 = fadd x86_fp80 %128, 0xKBFBCADC2C74C8B050000
  %136 = fadd x86_fp80 %134, %135
  %137 = bitcast x86_fp80 %129 to i80
  %138 = and i80 %137, -4294967296
  %139 = bitcast i80 %138 to x86_fp80
  %140 = fsub x86_fp80 %129, %139
  %141 = fmul x86_fp80 %15, %129
  %142 = fmul x86_fp80 %49, %139
  %143 = fsub x86_fp80 %142, %141
  %144 = fmul x86_fp80 %140, %49
  %145 = fadd x86_fp80 %144, %143
  %146 = fmul x86_fp80 %50, %139
  %147 = fadd x86_fp80 %146, %145
  %148 = fmul x86_fp80 %50, %140
  %149 = fadd x86_fp80 %148, %147
  %150 = fmul x86_fp80 %26, %129
  %151 = fadd x86_fp80 %150, %149
  %152 = fmul x86_fp80 %15, %136
  %153 = fadd x86_fp80 %151, %152
  %154 = fadd x86_fp80 %141, 0xK3FFF8000000000000000
  %155 = fsub x86_fp80 %154, %141
  %156 = fsub x86_fp80 %154, %155
  %157 = fsub x86_fp80 %141, %156
  %158 = fsub x86_fp80 0xK3FFF8000000000000000, %155
  %159 = fadd x86_fp80 %158, %157
  %160 = fadd x86_fp80 %159, %153
  %161 = fadd x86_fp80 %154, %160
  %162 = and i64 %11, 2
  %163 = icmp eq i64 %162, 0
  %164 = select i1 %163, x86_fp80 %161, x86_fp80 %91
  %165 = select i1 %163, x86_fp80 %91, x86_fp80 %161
  %166 = fcmp une x86_fp80 %1, 0xK7FFF8000000000000000
  %167 = fcmp une x86_fp80 %1, 0xKFFFF8000000000000000
  %168 = and i1 %167, %166
  br i1 %168, label %169, label %184

; <label>:169:                                    ; preds = %2
  %170 = add nsw i64 %12, 2
  %171 = and i64 %170, 4
  %172 = icmp eq i64 %171, 0
  %173 = fsub x86_fp80 0xK80000000000000000000, %164
  %174 = select i1 %172, x86_fp80 %164, x86_fp80 %173
  %175 = and i64 %11, 4
  %176 = icmp eq i64 %175, 0
  %177 = fsub x86_fp80 0xK80000000000000000000, %165
  %178 = select i1 %176, x86_fp80 %165, x86_fp80 %177
  %179 = fcmp oge x86_fp80 %1, 0xK00000000000000000000
  %180 = fsub x86_fp80 0xK80000000000000000000, %1
  %181 = select i1 %179, x86_fp80 %1, x86_fp80 %180
  %182 = fcmp ogt x86_fp80 %181, 0xK401CEE6B280000000000
  br i1 %182, label %183, label %184

; <label>:183:                                    ; preds = %169
  br label %184

; <label>:184:                                    ; preds = %2, %183, %169
  %185 = phi x86_fp80 [ 0xK7FFFC000000000000000, %2 ], [ 0xK00000000000000000000, %183 ], [ %174, %169 ]
  %186 = phi x86_fp80 [ 0xK7FFFC000000000000000, %2 ], [ 0xK00000000000000000000, %183 ], [ %178, %169 ]
  %187 = getelementptr inbounds %struct.Sleef_longdouble2, %struct.Sleef_longdouble2* %0, i64 0, i32 0
  store x86_fp80 %186, x86_fp80* %187, align 16
  %188 = bitcast %struct.Sleef_longdouble2* %0 to i8*
  %189 = getelementptr inbounds i8, i8* %188, i64 10
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %189, i8* nonnull %5, i64 6, i32 2, i1 false)
  %190 = getelementptr inbounds %struct.Sleef_longdouble2, %struct.Sleef_longdouble2* %0, i64 0, i32 1
  store x86_fp80 %185, x86_fp80* %190, align 16
  %191 = getelementptr inbounds i8, i8* %188, i64 26
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %191, i8* nonnull %6, i64 6, i32 2, i1 false)
  call void @llvm.lifetime.end.p0i8(i64 6, i8* nonnull %5)
  call void @llvm.lifetime.end.p0i8(i64 6, i8* nonnull %6)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define void @Sleef_sincospil_u35(%struct.Sleef_longdouble2* noalias nocapture sret, x86_fp80) local_unnamed_addr #0 {
  %3 = alloca [6 x i8], align 2
  %4 = alloca [6 x i8], align 2
  %5 = getelementptr inbounds [6 x i8], [6 x i8]* %3, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 6, i8* nonnull %5)
  %6 = getelementptr inbounds [6 x i8], [6 x i8]* %4, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 6, i8* nonnull %6)
  %7 = fmul x86_fp80 %1, 0xK40018000000000000000
  %8 = fptosi x86_fp80 %7 to i64
  %9 = fcmp uge x86_fp80 %7, 0xK00000000000000000000
  %10 = zext i1 %9 to i64
  %11 = add nsw i64 %10, %8
  %12 = and i64 %11, -2
  %13 = sitofp i64 %12 to x86_fp80
  %14 = fsub x86_fp80 %7, %13
  %15 = fmul x86_fp80 %14, %14
  %16 = fmul x86_fp80 %15, 0xK3FD1B63D9064D083A51C
  %17 = fsub x86_fp80 0xK3FD9F4779D47A193D307, %16
  %18 = fmul x86_fp80 %15, %17
  %19 = fadd x86_fp80 %18, 0xKBFE1F183A5EC173B4C3B
  %20 = fmul x86_fp80 %15, %19
  %21 = fadd x86_fp80 %20, 0xK3FE9A83C1A4310E5E5A9
  %22 = fmul x86_fp80 %15, %21
  %23 = fadd x86_fp80 %22, 0xKBFF099696673157C61C9
  %24 = fmul x86_fp80 %15, %23
  %25 = fadd x86_fp80 %24, 0xK3FF6A335E33BAD56D91F
  %26 = fmul x86_fp80 %15, %25
  %27 = fadd x86_fp80 %26, 0xKBFFBA55DE7312DF295E6
  %28 = fmul x86_fp80 %15, %27
  %29 = fadd x86_fp80 %28, 0xK3FFEC90FDAA22168C235
  %30 = fmul x86_fp80 %14, %29
  %31 = fmul x86_fp80 %15, 0xK3FCD8F27D840FBB1D713
  %32 = fadd x86_fp80 %31, 0xKBFD5DB6EFDD619B0D4C6
  %33 = fmul x86_fp80 %15, %32
  %34 = fadd x86_fp80 %33, 0xK3FDDFCE9C3127ADAF67B
  %35 = fmul x86_fp80 %15, %34
  %36 = fadd x86_fp80 %35, 0xKBFE5D368F94FE6E102E1
  %37 = fmul x86_fp80 %15, %36
  %38 = fadd x86_fp80 %37, 0xK3FECF0FA83448D22D7FC
  %39 = fmul x86_fp80 %15, %38
  %40 = fadd x86_fp80 %39, 0xKBFF3AAE9E3F1E5FF9196
  %41 = fmul x86_fp80 %15, %40
  %42 = fadd x86_fp80 %41, 0xK3FF981E0F840DAD61D7A
  %43 = fmul x86_fp80 %15, %42
  %44 = fadd x86_fp80 %43, 0xKBFFD9DE9E64DF22EF2D2
  %45 = fmul x86_fp80 %15, %44
  %46 = fadd x86_fp80 %45, 0xK3FFF8000000000000000
  %47 = and i64 %11, 2
  %48 = icmp eq i64 %47, 0
  %49 = select i1 %48, x86_fp80 %46, x86_fp80 %30
  %50 = select i1 %48, x86_fp80 %30, x86_fp80 %46
  %51 = fcmp une x86_fp80 %1, 0xK7FFF8000000000000000
  %52 = fcmp une x86_fp80 %1, 0xKFFFF8000000000000000
  %53 = and i1 %52, %51
  br i1 %53, label %54, label %69

; <label>:54:                                     ; preds = %2
  %55 = add nsw i64 %12, 2
  %56 = and i64 %55, 4
  %57 = icmp eq i64 %56, 0
  %58 = fsub x86_fp80 0xK80000000000000000000, %49
  %59 = select i1 %57, x86_fp80 %49, x86_fp80 %58
  %60 = and i64 %11, 4
  %61 = icmp eq i64 %60, 0
  %62 = fsub x86_fp80 0xK80000000000000000000, %50
  %63 = select i1 %61, x86_fp80 %50, x86_fp80 %62
  %64 = fcmp oge x86_fp80 %1, 0xK00000000000000000000
  %65 = fsub x86_fp80 0xK80000000000000000000, %1
  %66 = select i1 %64, x86_fp80 %1, x86_fp80 %65
  %67 = fcmp ogt x86_fp80 %66, 0xK401CEE6B280000000000
  br i1 %67, label %68, label %69

; <label>:68:                                     ; preds = %54
  br label %69

; <label>:69:                                     ; preds = %2, %68, %54
  %70 = phi x86_fp80 [ 0xK7FFFC000000000000000, %2 ], [ 0xK00000000000000000000, %68 ], [ %59, %54 ]
  %71 = phi x86_fp80 [ 0xK7FFFC000000000000000, %2 ], [ 0xK00000000000000000000, %68 ], [ %63, %54 ]
  %72 = getelementptr inbounds %struct.Sleef_longdouble2, %struct.Sleef_longdouble2* %0, i64 0, i32 0
  store x86_fp80 %71, x86_fp80* %72, align 16
  %73 = bitcast %struct.Sleef_longdouble2* %0 to i8*
  %74 = getelementptr inbounds i8, i8* %73, i64 10
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %74, i8* nonnull %5, i64 6, i32 2, i1 false)
  %75 = getelementptr inbounds %struct.Sleef_longdouble2, %struct.Sleef_longdouble2* %0, i64 0, i32 1
  store x86_fp80 %70, x86_fp80* %75, align 16
  %76 = getelementptr inbounds i8, i8* %73, i64 26
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %76, i8* nonnull %6, i64 6, i32 2, i1 false)
  call void @llvm.lifetime.end.p0i8(i64 6, i8* nonnull %5)
  call void @llvm.lifetime.end.p0i8(i64 6, i8* nonnull %6)
  ret void
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"}
