
bin/seq_bench/icc/matmul_O3:     file format elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
  401000:	48 83 ec 08          	sub    $0x8,%rsp
  401004:	48 8b 05 b5 5f 00 00 	mov    0x5fb5(%rip),%rax        # 406fc0 <__gmon_start__@Base>
  40100b:	48 85 c0             	test   %rax,%rax
  40100e:	74 02                	je     401012 <_init+0x12>
  401010:	ff d0                	call   *%rax
  401012:	48 83 c4 08          	add    $0x8,%rsp
  401016:	c3                   	ret

Disassembly of section .plt:

0000000000401020 <getenv@plt-0x10>:
  401020:	ff 35 ca 5f 00 00    	push   0x5fca(%rip)        # 406ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
  401026:	ff 25 cc 5f 00 00    	jmp    *0x5fcc(%rip)        # 406ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
  40102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401030 <getenv@plt>:
  401030:	ff 25 ca 5f 00 00    	jmp    *0x5fca(%rip)        # 407000 <getenv@GLIBC_2.2.5>
  401036:	68 00 00 00 00       	push   $0x0
  40103b:	e9 e0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401040 <free@plt>:
  401040:	ff 25 c2 5f 00 00    	jmp    *0x5fc2(%rip)        # 407008 <free@GLIBC_2.2.5>
  401046:	68 01 00 00 00       	push   $0x1
  40104b:	e9 d0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401050 <setenv@plt>:
  401050:	ff 25 ba 5f 00 00    	jmp    *0x5fba(%rip)        # 407010 <setenv@GLIBC_2.2.5>
  401056:	68 02 00 00 00       	push   $0x2
  40105b:	e9 c0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401060 <clock@plt>:
  401060:	ff 25 b2 5f 00 00    	jmp    *0x5fb2(%rip)        # 407018 <clock@GLIBC_2.2.5>
  401066:	68 03 00 00 00       	push   $0x3
  40106b:	e9 b0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401070 <fclose@plt>:
  401070:	ff 25 aa 5f 00 00    	jmp    *0x5faa(%rip)        # 407020 <fclose@GLIBC_2.2.5>
  401076:	68 04 00 00 00       	push   $0x4
  40107b:	e9 a0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401080 <strlen@plt>:
  401080:	ff 25 a2 5f 00 00    	jmp    *0x5fa2(%rip)        # 407028 <strlen@GLIBC_2.2.5>
  401086:	68 05 00 00 00       	push   $0x5
  40108b:	e9 90 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401090 <__stack_chk_fail@plt>:
  401090:	ff 25 9a 5f 00 00    	jmp    *0x5f9a(%rip)        # 407030 <__stack_chk_fail@GLIBC_2.4>
  401096:	68 06 00 00 00       	push   $0x6
  40109b:	e9 80 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010a0 <strchr@plt>:
  4010a0:	ff 25 92 5f 00 00    	jmp    *0x5f92(%rip)        # 407038 <strchr@GLIBC_2.2.5>
  4010a6:	68 07 00 00 00       	push   $0x7
  4010ab:	e9 70 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010b0 <memset@plt>:
  4010b0:	ff 25 8a 5f 00 00    	jmp    *0x5f8a(%rip)        # 407040 <memset@GLIBC_2.2.5>
  4010b6:	68 08 00 00 00       	push   $0x8
  4010bb:	e9 60 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010c0 <fputc@plt>:
  4010c0:	ff 25 82 5f 00 00    	jmp    *0x5f82(%rip)        # 407048 <fputc@GLIBC_2.2.5>
  4010c6:	68 09 00 00 00       	push   $0x9
  4010cb:	e9 50 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010d0 <fprintf@plt>:
  4010d0:	ff 25 7a 5f 00 00    	jmp    *0x5f7a(%rip)        # 407050 <fprintf@GLIBC_2.2.5>
  4010d6:	68 0a 00 00 00       	push   $0xa
  4010db:	e9 40 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010e0 <malloc@plt>:
  4010e0:	ff 25 72 5f 00 00    	jmp    *0x5f72(%rip)        # 407058 <malloc@GLIBC_2.2.5>
  4010e6:	68 0b 00 00 00       	push   $0xb
  4010eb:	e9 30 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010f0 <catopen@plt>:
  4010f0:	ff 25 6a 5f 00 00    	jmp    *0x5f6a(%rip)        # 407060 <catopen@GLIBC_2.2.5>
  4010f6:	68 0c 00 00 00       	push   $0xc
  4010fb:	e9 20 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401100 <__vsnprintf_chk@plt>:
  401100:	ff 25 62 5f 00 00    	jmp    *0x5f62(%rip)        # 407068 <__vsnprintf_chk@GLIBC_2.3.4>
  401106:	68 0d 00 00 00       	push   $0xd
  40110b:	e9 10 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401110 <__strncpy_chk@plt>:
  401110:	ff 25 5a 5f 00 00    	jmp    *0x5f5a(%rip)        # 407070 <__strncpy_chk@GLIBC_2.3.4>
  401116:	68 0e 00 00 00       	push   $0xe
  40111b:	e9 00 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401120 <__printf_chk@plt>:
  401120:	ff 25 52 5f 00 00    	jmp    *0x5f52(%rip)        # 407078 <__printf_chk@GLIBC_2.3.4>
  401126:	68 0f 00 00 00       	push   $0xf
  40112b:	e9 f0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401130 <fopen@plt>:
  401130:	ff 25 4a 5f 00 00    	jmp    *0x5f4a(%rip)        # 407080 <fopen@GLIBC_2.2.5>
  401136:	68 10 00 00 00       	push   $0x10
  40113b:	e9 e0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401140 <perror@plt>:
  401140:	ff 25 42 5f 00 00    	jmp    *0x5f42(%rip)        # 407088 <perror@GLIBC_2.2.5>
  401146:	68 11 00 00 00       	push   $0x11
  40114b:	e9 d0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401150 <catgets@plt>:
  401150:	ff 25 3a 5f 00 00    	jmp    *0x5f3a(%rip)        # 407090 <catgets@GLIBC_2.2.5>
  401156:	68 12 00 00 00       	push   $0x12
  40115b:	e9 c0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401160 <exit@plt>:
  401160:	ff 25 32 5f 00 00    	jmp    *0x5f32(%rip)        # 407098 <exit@GLIBC_2.2.5>
  401166:	68 13 00 00 00       	push   $0x13
  40116b:	e9 b0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401170 <__fprintf_chk@plt>:
  401170:	ff 25 2a 5f 00 00    	jmp    *0x5f2a(%rip)        # 4070a0 <__fprintf_chk@GLIBC_2.3.4>
  401176:	68 14 00 00 00       	push   $0x14
  40117b:	e9 a0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401180 <__strncat_chk@plt>:
  401180:	ff 25 22 5f 00 00    	jmp    *0x5f22(%rip)        # 4070a8 <__strncat_chk@GLIBC_2.3.4>
  401186:	68 15 00 00 00       	push   $0x15
  40118b:	e9 90 fe ff ff       	jmp    401020 <_init+0x20>

Disassembly of section .plt.got:

0000000000401190 <__cxa_finalize@plt>:
  401190:	ff 25 3a 5e 00 00    	jmp    *0x5e3a(%rip)        # 406fd0 <__cxa_finalize@GLIBC_2.2.5>
  401196:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004011a0 <_start>:
  4011a0:	31 ed                	xor    %ebp,%ebp
  4011a2:	49 89 d1             	mov    %rdx,%r9
  4011a5:	5e                   	pop    %rsi
  4011a6:	48 89 e2             	mov    %rsp,%rdx
  4011a9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4011ad:	50                   	push   %rax
  4011ae:	54                   	push   %rsp
  4011af:	45 31 c0             	xor    %r8d,%r8d
  4011b2:	31 c9                	xor    %ecx,%ecx
  4011b4:	48 c7 c7 90 12 40 00 	mov    $0x401290,%rdi
  4011bb:	ff 15 ef 5d 00 00    	call   *0x5def(%rip)        # 406fb0 <__libc_start_main@GLIBC_2.34>
  4011c1:	f4                   	hlt
  4011c2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4011c9:	00 00 00 
  4011cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004011d0 <deregister_tm_clones>:
  4011d0:	48 8d 3d e9 5e 00 00 	lea    0x5ee9(%rip),%rdi        # 4070c0 <stderr@GLIBC_2.2.5>
  4011d7:	48 8d 05 e2 5e 00 00 	lea    0x5ee2(%rip),%rax        # 4070c0 <stderr@GLIBC_2.2.5>
  4011de:	48 39 f8             	cmp    %rdi,%rax
  4011e1:	74 15                	je     4011f8 <deregister_tm_clones+0x28>
  4011e3:	48 8b 05 ce 5d 00 00 	mov    0x5dce(%rip),%rax        # 406fb8 <_ITM_deregisterTMCloneTable@Base>
  4011ea:	48 85 c0             	test   %rax,%rax
  4011ed:	74 09                	je     4011f8 <deregister_tm_clones+0x28>
  4011ef:	ff e0                	jmp    *%rax
  4011f1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4011f8:	c3                   	ret
  4011f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401200 <register_tm_clones>:
  401200:	48 8d 3d b9 5e 00 00 	lea    0x5eb9(%rip),%rdi        # 4070c0 <stderr@GLIBC_2.2.5>
  401207:	48 8d 35 b2 5e 00 00 	lea    0x5eb2(%rip),%rsi        # 4070c0 <stderr@GLIBC_2.2.5>
  40120e:	48 29 fe             	sub    %rdi,%rsi
  401211:	48 89 f0             	mov    %rsi,%rax
  401214:	48 c1 ee 3f          	shr    $0x3f,%rsi
  401218:	48 c1 f8 03          	sar    $0x3,%rax
  40121c:	48 01 c6             	add    %rax,%rsi
  40121f:	48 d1 fe             	sar    %rsi
  401222:	74 14                	je     401238 <register_tm_clones+0x38>
  401224:	48 8b 05 9d 5d 00 00 	mov    0x5d9d(%rip),%rax        # 406fc8 <_ITM_registerTMCloneTable@Base>
  40122b:	48 85 c0             	test   %rax,%rax
  40122e:	74 08                	je     401238 <register_tm_clones+0x38>
  401230:	ff e0                	jmp    *%rax
  401232:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401238:	c3                   	ret
  401239:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401240 <__do_global_dtors_aux>:
  401240:	f3 0f 1e fa          	endbr64
  401244:	80 3d 7d 5e 00 00 00 	cmpb   $0x0,0x5e7d(%rip)        # 4070c8 <completed.0>
  40124b:	75 2b                	jne    401278 <__do_global_dtors_aux+0x38>
  40124d:	55                   	push   %rbp
  40124e:	48 83 3d 7a 5d 00 00 	cmpq   $0x0,0x5d7a(%rip)        # 406fd0 <__cxa_finalize@GLIBC_2.2.5>
  401255:	00 
  401256:	48 89 e5             	mov    %rsp,%rbp
  401259:	74 0c                	je     401267 <__do_global_dtors_aux+0x27>
  40125b:	48 8b 3d 56 5e 00 00 	mov    0x5e56(%rip),%rdi        # 4070b8 <__dso_handle>
  401262:	e8 29 ff ff ff       	call   401190 <__cxa_finalize@plt>
  401267:	e8 64 ff ff ff       	call   4011d0 <deregister_tm_clones>
  40126c:	c6 05 55 5e 00 00 01 	movb   $0x1,0x5e55(%rip)        # 4070c8 <completed.0>
  401273:	5d                   	pop    %rbp
  401274:	c3                   	ret
  401275:	0f 1f 00             	nopl   (%rax)
  401278:	c3                   	ret
  401279:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401280 <frame_dummy>:
  401280:	f3 0f 1e fa          	endbr64
  401284:	e9 77 ff ff ff       	jmp    401200 <register_tm_clones>
  401289:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401290 <main>:
  401290:	55                   	push   %rbp
  401291:	48 89 e5             	mov    %rsp,%rbp
  401294:	48 83 e4 80          	and    $0xffffffffffffff80,%rsp
  401298:	41 54                	push   %r12
  40129a:	41 55                	push   %r13
  40129c:	41 56                	push   %r14
  40129e:	41 57                	push   %r15
  4012a0:	53                   	push   %rbx
  4012a1:	48 81 ec d8 00 00 00 	sub    $0xd8,%rsp
  4012a8:	bf 03 00 00 00       	mov    $0x3,%edi
  4012ad:	33 f6                	xor    %esi,%esi
  4012af:	e8 6c 0a 00 00       	call   401d20 <__intel_new_feature_proc_init>
  4012b4:	0f ae 1c 24          	stmxcsr (%rsp)
  4012b8:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4012bd:	81 0c 24 40 80 00 00 	orl    $0x8040,(%rsp)
  4012c4:	0f ae 14 24          	ldmxcsr (%rsp)
  4012c8:	e8 13 fe ff ff       	call   4010e0 <malloc@plt>
  4012cd:	49 89 c5             	mov    %rax,%r13
  4012d0:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4012d5:	e8 06 fe ff ff       	call   4010e0 <malloc@plt>
  4012da:	49 89 c4             	mov    %rax,%r12
  4012dd:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4012e2:	e8 f9 fd ff ff       	call   4010e0 <malloc@plt>
  4012e7:	48 89 c3             	mov    %rax,%rbx
  4012ea:	49 89 da             	mov    %rbx,%r10
  4012ed:	45 33 c0             	xor    %r8d,%r8d
  4012f0:	0f 10 15 19 2d 00 00 	movups 0x2d19(%rip),%xmm2        # 404010 <_IO_stdin_used+0x10>
  4012f7:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4012fb:	0f 10 0d 1e 2d 00 00 	movups 0x2d1e(%rip),%xmm1        # 404020 <_IO_stdin_used+0x20>
  401302:	48 be 00 00 00 00 00 	movabs $0x4000000000000000,%rsi
  401309:	00 00 40 
  40130c:	48 89 5c 24 30       	mov    %rbx,0x30(%rsp)
  401311:	49 b9 00 00 00 00 00 	movabs $0x4008000000000000,%r9
  401318:	00 08 40 
  40131b:	4c 89 a4 24 b0 00 00 	mov    %r12,0xb0(%rsp)
  401322:	00 
  401323:	4d 89 e6             	mov    %r12,%r14
  401326:	4c 89 6c 24 38       	mov    %r13,0x38(%rsp)
  40132b:	4d 89 eb             	mov    %r13,%r11
  40132e:	4d 89 d7             	mov    %r10,%r15
  401331:	33 c0                	xor    %eax,%eax
  401333:	4c 89 d3             	mov    %r10,%rbx
  401336:	48 83 e3 0f          	and    $0xf,%rbx
  40133a:	89 db                	mov    %ebx,%ebx
  40133c:	85 db                	test   %ebx,%ebx
  40133e:	74 17                	je     401357 <main+0xc7>
  401340:	f6 c3 07             	test   $0x7,%bl
  401343:	0f 85 5e 06 00 00    	jne    4019a7 <main+0x717>
  401349:	49 89 33             	mov    %rsi,(%r11)
  40134c:	bb 01 00 00 00       	mov    $0x1,%ebx
  401351:	4d 89 0e             	mov    %r9,(%r14)
  401354:	49 89 02             	mov    %rax,(%r10)
  401357:	89 da                	mov    %ebx,%edx
  401359:	f7 da                	neg    %edx
  40135b:	83 e2 07             	and    $0x7,%edx
  40135e:	f7 da                	neg    %edx
  401360:	81 c2 88 13 00 00    	add    $0x1388,%edx
  401366:	89 d1                	mov    %edx,%ecx
  401368:	49 8d 3c de          	lea    (%r14,%rbx,8),%rdi
  40136c:	48 f7 c7 0f 00 00 00 	test   $0xf,%rdi
  401373:	74 6c                	je     4013e1 <main+0x151>
  401375:	4d 8d 2c da          	lea    (%r10,%rbx,8),%r13
  401379:	4d 8d 24 de          	lea    (%r14,%rbx,8),%r12
  40137d:	49 8d 3c db          	lea    (%r11,%rbx,8),%rdi
  401381:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401388:	00 
  401389:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401390:	48 83 c3 08          	add    $0x8,%rbx
  401394:	0f 11 17             	movups %xmm2,(%rdi)
  401397:	41 0f 11 0c 24       	movups %xmm1,(%r12)
  40139c:	0f 11 57 10          	movups %xmm2,0x10(%rdi)
  4013a0:	41 0f 11 4c 24 10    	movups %xmm1,0x10(%r12)
  4013a6:	0f 11 57 20          	movups %xmm2,0x20(%rdi)
  4013aa:	41 0f 11 4c 24 20    	movups %xmm1,0x20(%r12)
  4013b0:	0f 11 57 30          	movups %xmm2,0x30(%rdi)
  4013b4:	41 0f 11 4c 24 30    	movups %xmm1,0x30(%r12)
  4013ba:	41 0f 11 45 00       	movups %xmm0,0x0(%r13)
  4013bf:	49 83 c4 40          	add    $0x40,%r12
  4013c3:	41 0f 11 45 10       	movups %xmm0,0x10(%r13)
  4013c8:	48 83 c7 40          	add    $0x40,%rdi
  4013cc:	41 0f 11 45 20       	movups %xmm0,0x20(%r13)
  4013d1:	41 0f 11 45 30       	movups %xmm0,0x30(%r13)
  4013d6:	49 83 c5 40          	add    $0x40,%r13
  4013da:	48 3b d9             	cmp    %rcx,%rbx
  4013dd:	72 b1                	jb     401390 <main+0x100>
  4013df:	eb 5e                	jmp    40143f <main+0x1af>
  4013e1:	4d 8d 2c da          	lea    (%r10,%rbx,8),%r13
  4013e5:	4d 8d 24 de          	lea    (%r14,%rbx,8),%r12
  4013e9:	49 8d 3c db          	lea    (%r11,%rbx,8),%rdi
  4013ed:	0f 1f 00             	nopl   (%rax)
  4013f0:	48 83 c3 08          	add    $0x8,%rbx
  4013f4:	0f 11 17             	movups %xmm2,(%rdi)
  4013f7:	0f 11 57 10          	movups %xmm2,0x10(%rdi)
  4013fb:	0f 11 57 20          	movups %xmm2,0x20(%rdi)
  4013ff:	0f 11 57 30          	movups %xmm2,0x30(%rdi)
  401403:	41 0f 11 0c 24       	movups %xmm1,(%r12)
  401408:	48 83 c7 40          	add    $0x40,%rdi
  40140c:	41 0f 11 45 00       	movups %xmm0,0x0(%r13)
  401411:	41 0f 11 4c 24 10    	movups %xmm1,0x10(%r12)
  401417:	41 0f 11 45 10       	movups %xmm0,0x10(%r13)
  40141c:	41 0f 11 4c 24 20    	movups %xmm1,0x20(%r12)
  401422:	41 0f 11 45 20       	movups %xmm0,0x20(%r13)
  401427:	41 0f 11 4c 24 30    	movups %xmm1,0x30(%r12)
  40142d:	49 83 c4 40          	add    $0x40,%r12
  401431:	41 0f 11 45 30       	movups %xmm0,0x30(%r13)
  401436:	49 83 c5 40          	add    $0x40,%r13
  40143a:	48 3b d9             	cmp    %rcx,%rbx
  40143d:	72 b1                	jb     4013f0 <main+0x160>
  40143f:	33 db                	xor    %ebx,%ebx
  401441:	8d 4a 01             	lea    0x1(%rdx),%ecx
  401444:	81 f9 88 13 00 00    	cmp    $0x1388,%ecx
  40144a:	77 22                	ja     40146e <main+0x1de>
  40144c:	89 d1                	mov    %edx,%ecx
  40144e:	f7 d9                	neg    %ecx
  401450:	81 c1 88 13 00 00    	add    $0x1388,%ecx
  401456:	8d 3c 1a             	lea    (%rdx,%rbx,1),%edi
  401459:	ff c3                	inc    %ebx
  40145b:	48 63 ff             	movslq %edi,%rdi
  40145e:	49 89 34 fb          	mov    %rsi,(%r11,%rdi,8)
  401462:	4d 89 0c fe          	mov    %r9,(%r14,%rdi,8)
  401466:	49 89 04 fa          	mov    %rax,(%r10,%rdi,8)
  40146a:	3b d9                	cmp    %ecx,%ebx
  40146c:	72 e8                	jb     401456 <main+0x1c6>
  40146e:	41 ff c0             	inc    %r8d
  401471:	49 81 c6 40 9c 00 00 	add    $0x9c40,%r14
  401478:	49 81 c3 40 9c 00 00 	add    $0x9c40,%r11
  40147f:	49 81 c2 40 9c 00 00 	add    $0x9c40,%r10
  401486:	41 81 f8 88 13 00 00 	cmp    $0x1388,%r8d
  40148d:	0f 82 a0 fe ff ff    	jb     401333 <main+0xa3>
  401493:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  401498:	4c 8b a4 24 b0 00 00 	mov    0xb0(%rsp),%r12
  40149f:	00 
  4014a0:	4c 8b 6c 24 38       	mov    0x38(%rsp),%r13
  4014a5:	e8 b6 fb ff ff       	call   401060 <clock@plt>
  4014aa:	49 89 c6             	mov    %rax,%r14
  4014ad:	4c 89 3c 24          	mov    %r15,(%rsp)
  4014b1:	45 33 c0             	xor    %r8d,%r8d
  4014b4:	4c 89 74 24 08       	mov    %r14,0x8(%rsp)
  4014b9:	be 80 00 00 00       	mov    $0x80,%esi
  4014be:	33 c0                	xor    %eax,%eax
  4014c0:	45 89 c2             	mov    %r8d,%r10d
  4014c3:	33 c9                	xor    %ecx,%ecx
  4014c5:	41 c1 e2 07          	shl    $0x7,%r10d
  4014c9:	49 69 d2 40 9c 00 00 	imul   $0x9c40,%r10,%rdx
  4014d0:	41 f7 da             	neg    %r10d
  4014d3:	4d 8d 4c 15 00       	lea    0x0(%r13,%rdx,1),%r9
  4014d8:	41 81 c2 88 13 00 00 	add    $0x1388,%r10d
  4014df:	48 03 d3             	add    %rbx,%rdx
  4014e2:	41 81 fa 80 00 00 00 	cmp    $0x80,%r10d
  4014e9:	44 89 44 24 28       	mov    %r8d,0x28(%rsp)
  4014ee:	44 0f 43 d6          	cmovae %esi,%r10d
  4014f2:	4d 63 d2             	movslq %r10d,%r10
  4014f5:	4c 89 54 24 40       	mov    %r10,0x40(%rsp)
  4014fa:	48 89 5c 24 30       	mov    %rbx,0x30(%rsp)
  4014ff:	4c 89 6c 24 38       	mov    %r13,0x38(%rsp)
  401504:	89 cb                	mov    %ecx,%ebx
  401506:	45 33 ff             	xor    %r15d,%r15d
  401509:	c1 e3 07             	shl    $0x7,%ebx
  40150c:	f7 db                	neg    %ebx
  40150e:	81 c3 88 13 00 00    	add    $0x1388,%ebx
  401514:	81 fb 80 00 00 00    	cmp    $0x80,%ebx
  40151a:	41 89 c8             	mov    %ecx,%r8d
  40151d:	0f 43 de             	cmovae %esi,%ebx
  401520:	49 c1 e0 0a          	shl    $0xa,%r8
  401524:	48 63 db             	movslq %ebx,%rbx
  401527:	4d 03 c1             	add    %r9,%r8
  40152a:	48 89 9c 24 80 00 00 	mov    %rbx,0x80(%rsp)
  401531:	00 
  401532:	4c 89 44 24 48       	mov    %r8,0x48(%rsp)
  401537:	4c 89 8c 24 88 00 00 	mov    %r9,0x88(%rsp)
  40153e:	00 
  40153f:	41 89 c8             	mov    %ecx,%r8d
  401542:	44 89 fb             	mov    %r15d,%ebx
  401545:	41 c1 e0 07          	shl    $0x7,%r8d
  401549:	49 89 c1             	mov    %rax,%r9
  40154c:	c1 e3 07             	shl    $0x7,%ebx
  40154f:	4d 89 ca             	mov    %r9,%r10
  401552:	4d 69 c0 40 9c 00 00 	imul   $0x9c40,%r8,%r8
  401559:	41 89 dd             	mov    %ebx,%r13d
  40155c:	4d 03 c4             	add    %r12,%r8
  40155f:	41 f7 dd             	neg    %r13d
  401562:	41 81 c5 88 13 00 00 	add    $0x1388,%r13d
  401569:	41 81 fd 80 00 00 00 	cmp    $0x80,%r13d
  401570:	45 89 fe             	mov    %r15d,%r14d
  401573:	44 0f 43 ee          	cmovae %esi,%r13d
  401577:	49 c1 e6 0a          	shl    $0xa,%r14
  40157b:	44 89 6c 24 18       	mov    %r13d,0x18(%rsp)
  401580:	89 5c 24 20          	mov    %ebx,0x20(%rsp)
  401584:	44 89 bc 24 a0 00 00 	mov    %r15d,0xa0(%rsp)
  40158b:	00 
  40158c:	41 8d 7d f9          	lea    -0x7(%r13),%edi
  401590:	89 7c 24 70          	mov    %edi,0x70(%rsp)
  401594:	4e 8d 1c 32          	lea    (%rdx,%r14,1),%r11
  401598:	4d 03 f0             	add    %r8,%r14
  40159b:	4c 89 74 24 10       	mov    %r14,0x10(%rsp)
  4015a0:	48 89 94 24 b8 00 00 	mov    %rdx,0xb8(%rsp)
  4015a7:	00 
  4015a8:	89 8c 24 a8 00 00 00 	mov    %ecx,0xa8(%rsp)
  4015af:	4c 89 a4 24 b0 00 00 	mov    %r12,0xb0(%rsp)
  4015b6:	00 
  4015b7:	4c 8b 6c 24 48       	mov    0x48(%rsp),%r13
  4015bc:	48 8b bc 24 b8 00 00 	mov    0xb8(%rsp),%rdi
  4015c3:	00 
  4015c4:	4f 8d 24 13          	lea    (%r11,%r10,1),%r12
  4015c8:	4c 89 e1             	mov    %r12,%rcx
  4015cb:	48 89 c3             	mov    %rax,%rbx
  4015ce:	48 83 e1 0f          	and    $0xf,%rcx
  4015d2:	4f 8d 74 15 00       	lea    0x0(%r13,%r10,1),%r14
  4015d7:	41 89 cf             	mov    %ecx,%r15d
  4015da:	49 03 fa             	add    %r10,%rdi
  4015dd:	41 83 e7 07          	and    $0x7,%r15d
  4015e1:	48 89 da             	mov    %rbx,%rdx
  4015e4:	4c 89 64 24 78       	mov    %r12,0x78(%rsp)
  4015e9:	4c 89 c6             	mov    %r8,%rsi
  4015ec:	4c 89 5c 24 58       	mov    %r11,0x58(%rsp)
  4015f1:	44 89 bc 24 98 00 00 	mov    %r15d,0x98(%rsp)
  4015f8:	00 
  4015f9:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
  4015fe:	4c 89 b4 24 90 00 00 	mov    %r14,0x90(%rsp)
  401605:	00 
  401606:	4c 89 54 24 60       	mov    %r10,0x60(%rsp)
  40160b:	4c 89 4c 24 50       	mov    %r9,0x50(%rsp)
  401610:	4c 8b 6c 24 10       	mov    0x10(%rsp),%r13
  401615:	44 8b 64 24 18       	mov    0x18(%rsp),%r12d
  40161a:	44 8b 5c 24 20       	mov    0x20(%rsp),%r11d
  40161f:	48 8b 84 24 90 00 00 	mov    0x90(%rsp),%rax
  401626:	00 
  401627:	f2 0f 10 0c d8       	movsd  (%rax,%rbx,8),%xmm1
  40162c:	41 83 fc 08          	cmp    $0x8,%r12d
  401630:	0f 82 45 03 00 00    	jb     40197b <main+0x6eb>
  401636:	89 c8                	mov    %ecx,%eax
  401638:	85 c9                	test   %ecx,%ecx
  40163a:	74 3b                	je     401677 <main+0x3e7>
  40163c:	83 bc 24 98 00 00 00 	cmpl   $0x0,0x98(%rsp)
  401643:	00 
  401644:	0f 85 31 03 00 00    	jne    40197b <main+0x6eb>
  40164a:	41 83 fc 09          	cmp    $0x9,%r12d
  40164e:	0f 82 27 03 00 00    	jb     40197b <main+0x6eb>
  401654:	f2 42 0f 10 04 2a    	movsd  (%rdx,%r13,1),%xmm0
  40165a:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  40165e:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  401663:	44 8b 74 24 70       	mov    0x70(%rsp),%r14d
  401668:	f2 0f 58 00          	addsd  (%rax),%xmm0
  40166c:	f2 0f 11 00          	movsd  %xmm0,(%rax)
  401670:	b8 01 00 00 00       	mov    $0x1,%eax
  401675:	eb 03                	jmp    40167a <main+0x3ea>
  401677:	45 89 e6             	mov    %r12d,%r14d
  40167a:	4d 63 ce             	movslq %r14d,%r9
  40167d:	46 8d 3c 18          	lea    (%rax,%r11,1),%r15d
  401681:	49 8d 3c 10          	lea    (%r8,%rdx,1),%rdi
  401685:	4a 8d 3c ff          	lea    (%rdi,%r15,8),%rdi
  401689:	4e 8d 14 2a          	lea    (%rdx,%r13,1),%r10
  40168d:	48 f7 c7 0f 00 00 00 	test   $0xf,%rdi
  401694:	74 76                	je     40170c <main+0x47c>
  401696:	0f 28 c1             	movaps %xmm1,%xmm0
  401699:	89 c0                	mov    %eax,%eax
  40169b:	66 0f 14 c0          	unpcklpd %xmm0,%xmm0
  40169f:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
  4016a4:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4016a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4016b0:	41 0f 10 14 c2       	movups (%r10,%rax,8),%xmm2
  4016b5:	41 0f 10 5c c2 10    	movups 0x10(%r10,%rax,8),%xmm3
  4016bb:	41 0f 10 64 c2 20    	movups 0x20(%r10,%rax,8),%xmm4
  4016c1:	41 0f 10 6c c2 30    	movups 0x30(%r10,%rax,8),%xmm5
  4016c7:	66 0f 59 d0          	mulpd  %xmm0,%xmm2
  4016cb:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  4016cf:	66 0f 59 e0          	mulpd  %xmm0,%xmm4
  4016d3:	66 0f 59 e8          	mulpd  %xmm0,%xmm5
  4016d7:	66 0f 58 14 c7       	addpd  (%rdi,%rax,8),%xmm2
  4016dc:	66 0f 58 5c c7 10    	addpd  0x10(%rdi,%rax,8),%xmm3
  4016e2:	66 0f 58 64 c7 20    	addpd  0x20(%rdi,%rax,8),%xmm4
  4016e8:	66 0f 58 6c c7 30    	addpd  0x30(%rdi,%rax,8),%xmm5
  4016ee:	0f 11 14 c7          	movups %xmm2,(%rdi,%rax,8)
  4016f2:	0f 11 5c c7 10       	movups %xmm3,0x10(%rdi,%rax,8)
  4016f7:	0f 11 64 c7 20       	movups %xmm4,0x20(%rdi,%rax,8)
  4016fc:	0f 11 6c c7 30       	movups %xmm5,0x30(%rdi,%rax,8)
  401701:	48 83 c0 08          	add    $0x8,%rax
  401705:	49 3b c1             	cmp    %r9,%rax
  401708:	72 a6                	jb     4016b0 <main+0x420>
  40170a:	eb 6e                	jmp    40177a <main+0x4ea>
  40170c:	0f 28 c1             	movaps %xmm1,%xmm0
  40170f:	66 0f 14 c0          	unpcklpd %xmm0,%xmm0
  401713:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
  401718:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40171f:	00 
  401720:	41 0f 10 14 c2       	movups (%r10,%rax,8),%xmm2
  401725:	41 0f 10 5c c2 10    	movups 0x10(%r10,%rax,8),%xmm3
  40172b:	41 0f 10 64 c2 20    	movups 0x20(%r10,%rax,8),%xmm4
  401731:	41 0f 10 6c c2 30    	movups 0x30(%r10,%rax,8),%xmm5
  401737:	66 0f 59 d0          	mulpd  %xmm0,%xmm2
  40173b:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  40173f:	66 0f 59 e0          	mulpd  %xmm0,%xmm4
  401743:	66 0f 59 e8          	mulpd  %xmm0,%xmm5
  401747:	66 0f 58 14 c7       	addpd  (%rdi,%rax,8),%xmm2
  40174c:	66 0f 58 5c c7 10    	addpd  0x10(%rdi,%rax,8),%xmm3
  401752:	66 0f 58 64 c7 20    	addpd  0x20(%rdi,%rax,8),%xmm4
  401758:	66 0f 58 6c c7 30    	addpd  0x30(%rdi,%rax,8),%xmm5
  40175e:	0f 11 14 c7          	movups %xmm2,(%rdi,%rax,8)
  401762:	0f 11 5c c7 10       	movups %xmm3,0x10(%rdi,%rax,8)
  401767:	0f 11 64 c7 20       	movups %xmm4,0x20(%rdi,%rax,8)
  40176c:	0f 11 6c c7 30       	movups %xmm5,0x30(%rdi,%rax,8)
  401771:	48 83 c0 08          	add    $0x8,%rax
  401775:	49 3b c1             	cmp    %r9,%rax
  401778:	72 a6                	jb     401720 <main+0x490>
  40177a:	33 c0                	xor    %eax,%eax
  40177c:	41 8d 7e 01          	lea    0x1(%r14),%edi
  401780:	41 3b fc             	cmp    %r12d,%edi
  401783:	77 39                	ja     4017be <main+0x52e>
  401785:	4c 8b 54 24 68       	mov    0x68(%rsp),%r10
  40178a:	43 8d 3c 1e          	lea    (%r14,%r11,1),%edi
  40178e:	48 63 ff             	movslq %edi,%rdi
  401791:	41 f7 de             	neg    %r14d
  401794:	45 03 f4             	add    %r12d,%r14d
  401797:	4d 63 f6             	movslq %r14d,%r14
  40179a:	4c 8d 0c fe          	lea    (%rsi,%rdi,8),%r9
  40179e:	49 8d 3c fa          	lea    (%r10,%rdi,8),%rdi
  4017a2:	f2 41 0f 10 04 c1    	movsd  (%r9,%rax,8),%xmm0
  4017a8:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  4017ac:	f2 0f 58 04 c7       	addsd  (%rdi,%rax,8),%xmm0
  4017b1:	f2 0f 11 04 c7       	movsd  %xmm0,(%rdi,%rax,8)
  4017b6:	48 ff c0             	inc    %rax
  4017b9:	49 3b c6             	cmp    %r14,%rax
  4017bc:	72 e4                	jb     4017a2 <main+0x512>
  4017be:	48 ff c3             	inc    %rbx
  4017c1:	48 81 c6 40 9c 00 00 	add    $0x9c40,%rsi
  4017c8:	48 81 c2 40 9c 00 00 	add    $0x9c40,%rdx
  4017cf:	48 3b 9c 24 80 00 00 	cmp    0x80(%rsp),%rbx
  4017d6:	00 
  4017d7:	0f 82 42 fe ff ff    	jb     40161f <main+0x38f>
  4017dd:	4c 8b 4c 24 50       	mov    0x50(%rsp),%r9
  4017e2:	33 c0                	xor    %eax,%eax
  4017e4:	49 ff c1             	inc    %r9
  4017e7:	4c 8b 54 24 60       	mov    0x60(%rsp),%r10
  4017ec:	49 81 c2 40 9c 00 00 	add    $0x9c40,%r10
  4017f3:	4c 8b 5c 24 58       	mov    0x58(%rsp),%r11
  4017f8:	4c 8b 6c 24 48       	mov    0x48(%rsp),%r13
  4017fd:	4c 3b 4c 24 40       	cmp    0x40(%rsp),%r9
  401802:	0f 82 b4 fd ff ff    	jb     4015bc <main+0x32c>
  401808:	44 8b bc 24 a0 00 00 	mov    0xa0(%rsp),%r15d
  40180f:	00 
  401810:	be 80 00 00 00       	mov    $0x80,%esi
  401815:	41 ff c7             	inc    %r15d
  401818:	48 8b 94 24 b8 00 00 	mov    0xb8(%rsp),%rdx
  40181f:	00 
  401820:	8b 8c 24 a8 00 00 00 	mov    0xa8(%rsp),%ecx
  401827:	4c 8b a4 24 b0 00 00 	mov    0xb0(%rsp),%r12
  40182e:	00 
  40182f:	41 83 ff 28          	cmp    $0x28,%r15d
  401833:	0f 82 06 fd ff ff    	jb     40153f <main+0x2af>
  401839:	ff c1                	inc    %ecx
  40183b:	4c 8b 8c 24 88 00 00 	mov    0x88(%rsp),%r9
  401842:	00 
  401843:	83 f9 28             	cmp    $0x28,%ecx
  401846:	0f 82 b8 fc ff ff    	jb     401504 <main+0x274>
  40184c:	44 8b 44 24 28       	mov    0x28(%rsp),%r8d
  401851:	41 ff c0             	inc    %r8d
  401854:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  401859:	4c 8b 6c 24 38       	mov    0x38(%rsp),%r13
  40185e:	41 83 f8 28          	cmp    $0x28,%r8d
  401862:	0f 82 58 fc ff ff    	jb     4014c0 <main+0x230>
  401868:	4c 8b 3c 24          	mov    (%rsp),%r15
  40186c:	4c 8b 74 24 08       	mov    0x8(%rsp),%r14
  401871:	e8 ea f7 ff ff       	call   401060 <clock@plt>
  401876:	49 2b c6             	sub    %r14,%rax
  401879:	66 0f ef c0          	pxor   %xmm0,%xmm0
  40187d:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
  401882:	f2 0f 5e 05 b6 27 00 	divsd  0x27b6(%rip),%xmm0        # 404040 <_IO_stdin_used+0x40>
  401889:	00 
  40188a:	be 50 40 40 00       	mov    $0x404050,%esi
  40188f:	ba 88 13 00 00       	mov    $0x1388,%edx
  401894:	b8 01 00 00 00       	mov    $0x1,%eax
  401899:	48 8b 3d 20 58 00 00 	mov    0x5820(%rip),%rdi        # 4070c0 <stderr@GLIBC_2.2.5>
  4018a0:	e8 2b f8 ff ff       	call   4010d0 <fprintf@plt>
  4018a5:	bf 6c 40 40 00       	mov    $0x40406c,%edi
  4018aa:	be 78 40 40 00       	mov    $0x404078,%esi
  4018af:	e8 7c f8 ff ff       	call   401130 <fopen@plt>
  4018b4:	49 89 c6             	mov    %rax,%r14
  4018b7:	4d 85 f6             	test   %r14,%r14
  4018ba:	0f 84 c3 00 00 00    	je     401983 <main+0x6f3>
  4018c0:	4c 89 f7             	mov    %r14,%rdi
  4018c3:	be 7c 40 40 00       	mov    $0x40407c,%esi
  4018c8:	ba 88 13 00 00       	mov    $0x1388,%edx
  4018cd:	33 c0                	xor    %eax,%eax
  4018cf:	e8 fc f7 ff ff       	call   4010d0 <fprintf@plt>
  4018d4:	33 c0                	xor    %eax,%eax
  4018d6:	48 89 5c 24 30       	mov    %rbx,0x30(%rsp)
  4018db:	89 c3                	mov    %eax,%ebx
  4018dd:	4c 89 a4 24 b0 00 00 	mov    %r12,0xb0(%rsp)
  4018e4:	00 
  4018e5:	4c 89 6c 24 38       	mov    %r13,0x38(%rsp)
  4018ea:	45 33 ed             	xor    %r13d,%r13d
  4018ed:	4d 89 ec             	mov    %r13,%r12
  4018f0:	f2 43 0f 10 04 e7    	movsd  (%r15,%r12,8),%xmm0
  4018f6:	4c 89 f7             	mov    %r14,%rdi
  4018f9:	be 84 40 40 00       	mov    $0x404084,%esi
  4018fe:	b8 01 00 00 00       	mov    $0x1,%eax
  401903:	e8 c8 f7 ff ff       	call   4010d0 <fprintf@plt>
  401908:	49 ff c4             	inc    %r12
  40190b:	49 81 fc e8 03 00 00 	cmp    $0x3e8,%r12
  401912:	7c dc                	jl     4018f0 <main+0x660>
  401914:	bf 0a 00 00 00       	mov    $0xa,%edi
  401919:	4c 89 f6             	mov    %r14,%rsi
  40191c:	e8 9f f7 ff ff       	call   4010c0 <fputc@plt>
  401921:	ff c3                	inc    %ebx
  401923:	49 81 c7 40 9c 00 00 	add    $0x9c40,%r15
  40192a:	81 fb e8 03 00 00    	cmp    $0x3e8,%ebx
  401930:	7c bb                	jl     4018ed <main+0x65d>
  401932:	4c 89 f7             	mov    %r14,%rdi
  401935:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  40193a:	4c 8b a4 24 b0 00 00 	mov    0xb0(%rsp),%r12
  401941:	00 
  401942:	4c 8b 6c 24 38       	mov    0x38(%rsp),%r13
  401947:	e8 24 f7 ff ff       	call   401070 <fclose@plt>
  40194c:	4c 89 ef             	mov    %r13,%rdi
  40194f:	e8 ec f6 ff ff       	call   401040 <free@plt>
  401954:	4c 89 e7             	mov    %r12,%rdi
  401957:	e8 e4 f6 ff ff       	call   401040 <free@plt>
  40195c:	48 89 df             	mov    %rbx,%rdi
  40195f:	e8 dc f6 ff ff       	call   401040 <free@plt>
  401964:	33 c0                	xor    %eax,%eax
  401966:	48 81 c4 d8 00 00 00 	add    $0xd8,%rsp
  40196d:	5b                   	pop    %rbx
  40196e:	41 5f                	pop    %r15
  401970:	41 5e                	pop    %r14
  401972:	41 5d                	pop    %r13
  401974:	41 5c                	pop    %r12
  401976:	48 89 ec             	mov    %rbp,%rsp
  401979:	5d                   	pop    %rbp
  40197a:	c3                   	ret
  40197b:	45 33 f6             	xor    %r14d,%r14d
  40197e:	e9 f7 fd ff ff       	jmp    40177a <main+0x4ea>
  401983:	bf 8c 40 40 00       	mov    $0x40408c,%edi
  401988:	e8 b3 f7 ff ff       	call   401140 <perror@plt>
  40198d:	b8 01 00 00 00       	mov    $0x1,%eax
  401992:	48 81 c4 d8 00 00 00 	add    $0xd8,%rsp
  401999:	5b                   	pop    %rbx
  40199a:	41 5f                	pop    %r15
  40199c:	41 5e                	pop    %r14
  40199e:	41 5d                	pop    %r13
  4019a0:	41 5c                	pop    %r12
  4019a2:	48 89 ec             	mov    %rbp,%rsp
  4019a5:	5d                   	pop    %rbp
  4019a6:	c3                   	ret
  4019a7:	33 d2                	xor    %edx,%edx
  4019a9:	e9 91 fa ff ff       	jmp    40143f <main+0x1af>
  4019ae:	66 90                	xchg   %ax,%ax

00000000004019b0 <__intel_new_feature_proc_init_n>:
  4019b0:	f3 0f 1e fa          	endbr64
  4019b4:	55                   	push   %rbp
  4019b5:	41 57                	push   %r15
  4019b7:	41 56                	push   %r14
  4019b9:	41 55                	push   %r13
  4019bb:	41 54                	push   %r12
  4019bd:	53                   	push   %rbx
  4019be:	48 81 ec 38 04 00 00 	sub    $0x438,%rsp
  4019c5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4019cc:	00 00 
  4019ce:	48 89 84 24 30 04 00 	mov    %rax,0x430(%rsp)
  4019d5:	00 
  4019d6:	0f 57 c0             	xorps  %xmm0,%xmm0
  4019d9:	0f 29 44 24 10       	movaps %xmm0,0x10(%rsp)
  4019de:	48 c7 c1 d0 70 40 00 	mov    $0x4070d0,%rcx
  4019e5:	48 83 39 00          	cmpq   $0x0,(%rcx)
  4019e9:	75 17                	jne    401a02 <__intel_new_feature_proc_init_n+0x52>
  4019eb:	e8 80 04 00 00       	call   401e70 <__intel_cpu_features_init>
  4019f0:	85 c0                	test   %eax,%eax
  4019f2:	0f 85 0b 02 00 00    	jne    401c03 <__intel_new_feature_proc_init_n+0x253>
  4019f8:	48 83 39 00          	cmpq   $0x0,(%rcx)
  4019fc:	0f 84 01 02 00 00    	je     401c03 <__intel_new_feature_proc_init_n+0x253>
  401a02:	83 ff 02             	cmp    $0x2,%edi
  401a05:	7d 38                	jge    401a3f <__intel_new_feature_proc_init_n+0x8f>
  401a07:	48 63 c7             	movslq %edi,%rax
  401a0a:	48 8b 0c c1          	mov    (%rcx,%rax,8),%rcx
  401a0e:	48 f7 d1             	not    %rcx
  401a11:	48 85 ce             	test   %rcx,%rsi
  401a14:	75 48                	jne    401a5e <__intel_new_feature_proc_init_n+0xae>
  401a16:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401a1d:	00 00 
  401a1f:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  401a26:	00 
  401a27:	0f 85 e8 02 00 00    	jne    401d15 <__intel_new_feature_proc_init_n+0x365>
  401a2d:	48 81 c4 38 04 00 00 	add    $0x438,%rsp
  401a34:	5b                   	pop    %rbx
  401a35:	41 5c                	pop    %r12
  401a37:	41 5d                	pop    %r13
  401a39:	41 5e                	pop    %r14
  401a3b:	41 5f                	pop    %r15
  401a3d:	5d                   	pop    %rbp
  401a3e:	c3                   	ret
  401a3f:	bf 01 00 00 00       	mov    $0x1,%edi
  401a44:	31 f6                	xor    %esi,%esi
  401a46:	31 d2                	xor    %edx,%edx
  401a48:	31 c0                	xor    %eax,%eax
  401a4a:	e8 b1 1e 00 00       	call   403900 <__libirc_print>
  401a4f:	bf 01 00 00 00       	mov    $0x1,%edi
  401a54:	be 3a 00 00 00       	mov    $0x3a,%esi
  401a59:	e9 bf 01 00 00       	jmp    401c1d <__intel_new_feature_proc_init_n+0x26d>
  401a5e:	48 21 f1             	and    %rsi,%rcx
  401a61:	48 89 4c c4 10       	mov    %rcx,0x10(%rsp,%rax,8)
  401a66:	45 31 ff             	xor    %r15d,%r15d
  401a69:	bf 39 00 00 00       	mov    $0x39,%edi
  401a6e:	31 f6                	xor    %esi,%esi
  401a70:	31 c0                	xor    %eax,%eax
  401a72:	e8 49 1c 00 00       	call   4036c0 <__libirc_get_msg>
  401a77:	48 89 04 24          	mov    %rax,(%rsp)
  401a7b:	c6 44 24 30 00       	movb   $0x0,0x30(%rsp)
  401a80:	bd 01 00 00 00       	mov    $0x1,%ebp
  401a85:	4c 8d 64 24 10       	lea    0x10(%rsp),%r12
  401a8a:	4c 8d 6c 24 30       	lea    0x30(%rsp),%r13
  401a8f:	31 db                	xor    %ebx,%ebx
  401a91:	eb 31                	jmp    401ac4 <__intel_new_feature_proc_init_n+0x114>
  401a93:	b8 ff 03 00 00       	mov    $0x3ff,%eax
  401a98:	44 29 f8             	sub    %r15d,%eax
  401a9b:	48 63 d0             	movslq %eax,%rdx
  401a9e:	b9 00 04 00 00       	mov    $0x400,%ecx
  401aa3:	4c 89 ef             	mov    %r13,%rdi
  401aa6:	4c 89 f6             	mov    %r14,%rsi
  401aa9:	e8 d2 f6 ff ff       	call   401180 <__strncat_chk@plt>
  401aae:	4c 89 ef             	mov    %r13,%rdi
  401ab1:	e8 ca f5 ff ff       	call   401080 <strlen@plt>
  401ab6:	49 89 c7             	mov    %rax,%r15
  401ab9:	ff c5                	inc    %ebp
  401abb:	83 fd 47             	cmp    $0x47,%ebp
  401abe:	0f 84 26 01 00 00    	je     401bea <__intel_new_feature_proc_init_n+0x23a>
  401ac4:	89 e8                	mov    %ebp,%eax
  401ac6:	e8 95 19 00 00       	call   403460 <__libirc_get_feature_bitpos>
  401acb:	85 c0                	test   %eax,%eax
  401acd:	78 ea                	js     401ab9 <__intel_new_feature_proc_init_n+0x109>
  401acf:	4c 89 e7             	mov    %r12,%rdi
  401ad2:	89 ee                	mov    %ebp,%esi
  401ad4:	e8 c7 19 00 00       	call   4034a0 <__libirc_get_cpu_feature>
  401ad9:	85 c0                	test   %eax,%eax
  401adb:	74 dc                	je     401ab9 <__intel_new_feature_proc_init_n+0x109>
  401add:	4c 89 e7             	mov    %r12,%rdi
  401ae0:	89 ee                	mov    %ebp,%esi
  401ae2:	e8 b9 19 00 00       	call   4034a0 <__libirc_get_cpu_feature>
  401ae7:	85 c0                	test   %eax,%eax
  401ae9:	0f 88 e6 00 00 00    	js     401bd5 <__intel_new_feature_proc_init_n+0x225>
  401aef:	89 ef                	mov    %ebp,%edi
  401af1:	e8 7a 0e 00 00       	call   402970 <__libirc_get_feature_name>
  401af6:	48 85 c0             	test   %rax,%rax
  401af9:	0f 84 d6 00 00 00    	je     401bd5 <__intel_new_feature_proc_init_n+0x225>
  401aff:	49 89 c6             	mov    %rax,%r14
  401b02:	80 38 00             	cmpb   $0x0,(%rax)
  401b05:	0f 84 ca 00 00 00    	je     401bd5 <__intel_new_feature_proc_init_n+0x225>
  401b0b:	80 7c 24 30 00       	cmpb   $0x0,0x30(%rsp)
  401b10:	74 81                	je     401a93 <__intel_new_feature_proc_init_n+0xe3>
  401b12:	48 85 db             	test   %rbx,%rbx
  401b15:	0f 84 b2 00 00 00    	je     401bcd <__intel_new_feature_proc_init_n+0x21d>
  401b1b:	4d 89 ec             	mov    %r13,%r12
  401b1e:	48 89 df             	mov    %rbx,%rdi
  401b21:	e8 5a f5 ff ff       	call   401080 <strlen@plt>
  401b26:	49 89 c5             	mov    %rax,%r13
  401b29:	48 8d 3d 64 25 00 00 	lea    0x2564(%rip),%rdi        # 404094 <_IO_stdin_used+0x94>
  401b30:	e8 4b f5 ff ff       	call   401080 <strlen@plt>
  401b35:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  401b3a:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  401b3f:	49 63 df             	movslq %r15d,%rbx
  401b42:	48 8b 3c 24          	mov    (%rsp),%rdi
  401b46:	e8 35 f5 ff ff       	call   401080 <strlen@plt>
  401b4b:	49 89 c7             	mov    %rax,%r15
  401b4e:	4c 89 f7             	mov    %r14,%rdi
  401b51:	e8 2a f5 ff ff       	call   401080 <strlen@plt>
  401b56:	49 01 dd             	add    %rbx,%r13
  401b59:	4c 03 6c 24 28       	add    0x28(%rsp),%r13
  401b5e:	4c 01 f8             	add    %r15,%rax
  401b61:	4c 01 e8             	add    %r13,%rax
  401b64:	b9 ff 03 00 00       	mov    $0x3ff,%ecx
  401b69:	29 d9                	sub    %ebx,%ecx
  401b6b:	48 63 d1             	movslq %ecx,%rdx
  401b6e:	48 3d ff 03 00 00    	cmp    $0x3ff,%rax
  401b74:	0f 87 dd 00 00 00    	ja     401c57 <__intel_new_feature_proc_init_n+0x2a7>
  401b7a:	b9 00 04 00 00       	mov    $0x400,%ecx
  401b7f:	4d 89 e5             	mov    %r12,%r13
  401b82:	4c 89 e7             	mov    %r12,%rdi
  401b85:	48 8d 35 08 25 00 00 	lea    0x2508(%rip),%rsi        # 404094 <_IO_stdin_used+0x94>
  401b8c:	e8 ef f5 ff ff       	call   401180 <__strncat_chk@plt>
  401b91:	4c 89 e7             	mov    %r12,%rdi
  401b94:	e8 e7 f4 ff ff       	call   401080 <strlen@plt>
  401b99:	48 c1 e0 20          	shl    $0x20,%rax
  401b9d:	48 ba 00 00 00 00 ff 	movabs $0x3ff00000000,%rdx
  401ba4:	03 00 00 
  401ba7:	48 29 c2             	sub    %rax,%rdx
  401baa:	48 c1 fa 20          	sar    $0x20,%rdx
  401bae:	b9 00 04 00 00       	mov    $0x400,%ecx
  401bb3:	4c 89 e7             	mov    %r12,%rdi
  401bb6:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  401bbb:	e8 c0 f5 ff ff       	call   401180 <__strncat_chk@plt>
  401bc0:	4c 89 f3             	mov    %r14,%rbx
  401bc3:	4c 8d 64 24 10       	lea    0x10(%rsp),%r12
  401bc8:	e9 e1 fe ff ff       	jmp    401aae <__intel_new_feature_proc_init_n+0xfe>
  401bcd:	4c 89 f3             	mov    %r14,%rbx
  401bd0:	e9 e4 fe ff ff       	jmp    401ab9 <__intel_new_feature_proc_init_n+0x109>
  401bd5:	bf 01 00 00 00       	mov    $0x1,%edi
  401bda:	31 f6                	xor    %esi,%esi
  401bdc:	31 d2                	xor    %edx,%edx
  401bde:	31 c0                	xor    %eax,%eax
  401be0:	e8 1b 1d 00 00       	call   403900 <__libirc_print>
  401be5:	e9 ce 00 00 00       	jmp    401cb8 <__intel_new_feature_proc_init_n+0x308>
  401bea:	48 85 db             	test   %rbx,%rbx
  401bed:	0f 84 ac 00 00 00    	je     401c9f <__intel_new_feature_proc_init_n+0x2ef>
  401bf3:	49 89 dc             	mov    %rbx,%r12
  401bf6:	b8 ff 03 00 00       	mov    $0x3ff,%eax
  401bfb:	44 29 f8             	sub    %r15d,%eax
  401bfe:	48 63 d0             	movslq %eax,%rdx
  401c01:	eb 59                	jmp    401c5c <__intel_new_feature_proc_init_n+0x2ac>
  401c03:	bf 01 00 00 00       	mov    $0x1,%edi
  401c08:	31 f6                	xor    %esi,%esi
  401c0a:	31 d2                	xor    %edx,%edx
  401c0c:	31 c0                	xor    %eax,%eax
  401c0e:	e8 ed 1c 00 00       	call   403900 <__libirc_print>
  401c13:	bf 01 00 00 00       	mov    $0x1,%edi
  401c18:	be 3b 00 00 00       	mov    $0x3b,%esi
  401c1d:	31 d2                	xor    %edx,%edx
  401c1f:	31 c0                	xor    %eax,%eax
  401c21:	e8 da 1c 00 00       	call   403900 <__libirc_print>
  401c26:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401c2d:	00 00 
  401c2f:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  401c36:	00 
  401c37:	0f 85 d8 00 00 00    	jne    401d15 <__intel_new_feature_proc_init_n+0x365>
  401c3d:	bf 01 00 00 00       	mov    $0x1,%edi
  401c42:	31 f6                	xor    %esi,%esi
  401c44:	31 d2                	xor    %edx,%edx
  401c46:	31 c0                	xor    %eax,%eax
  401c48:	e8 b3 1c 00 00       	call   403900 <__libirc_print>
  401c4d:	bf 01 00 00 00       	mov    $0x1,%edi
  401c52:	e8 09 f5 ff ff       	call   401160 <exit@plt>
  401c57:	4c 8b 64 24 08       	mov    0x8(%rsp),%r12
  401c5c:	4c 8d 74 24 30       	lea    0x30(%rsp),%r14
  401c61:	b9 00 04 00 00       	mov    $0x400,%ecx
  401c66:	4c 89 f7             	mov    %r14,%rdi
  401c69:	48 8b 34 24          	mov    (%rsp),%rsi
  401c6d:	e8 0e f5 ff ff       	call   401180 <__strncat_chk@plt>
  401c72:	4c 89 f7             	mov    %r14,%rdi
  401c75:	e8 06 f4 ff ff       	call   401080 <strlen@plt>
  401c7a:	48 c1 e0 20          	shl    $0x20,%rax
  401c7e:	48 ba 00 00 00 00 ff 	movabs $0x3ff00000000,%rdx
  401c85:	03 00 00 
  401c88:	48 29 c2             	sub    %rax,%rdx
  401c8b:	48 c1 fa 20          	sar    $0x20,%rdx
  401c8f:	b9 00 04 00 00       	mov    $0x400,%ecx
  401c94:	4c 89 f7             	mov    %r14,%rdi
  401c97:	4c 89 e6             	mov    %r12,%rsi
  401c9a:	e8 e1 f4 ff ff       	call   401180 <__strncat_chk@plt>
  401c9f:	0f b6 5c 24 30       	movzbl 0x30(%rsp),%ebx
  401ca4:	bf 01 00 00 00       	mov    $0x1,%edi
  401ca9:	31 f6                	xor    %esi,%esi
  401cab:	31 d2                	xor    %edx,%edx
  401cad:	31 c0                	xor    %eax,%eax
  401caf:	e8 4c 1c 00 00       	call   403900 <__libirc_print>
  401cb4:	84 db                	test   %bl,%bl
  401cb6:	75 15                	jne    401ccd <__intel_new_feature_proc_init_n+0x31d>
  401cb8:	bf 01 00 00 00       	mov    $0x1,%edi
  401cbd:	be 3a 00 00 00       	mov    $0x3a,%esi
  401cc2:	31 d2                	xor    %edx,%edx
  401cc4:	31 c0                	xor    %eax,%eax
  401cc6:	e8 35 1c 00 00       	call   403900 <__libirc_print>
  401ccb:	eb 1b                	jmp    401ce8 <__intel_new_feature_proc_init_n+0x338>
  401ccd:	48 8d 4c 24 30       	lea    0x30(%rsp),%rcx
  401cd2:	bf 01 00 00 00       	mov    $0x1,%edi
  401cd7:	be 38 00 00 00       	mov    $0x38,%esi
  401cdc:	ba 01 00 00 00       	mov    $0x1,%edx
  401ce1:	31 c0                	xor    %eax,%eax
  401ce3:	e8 18 1c 00 00       	call   403900 <__libirc_print>
  401ce8:	bf 01 00 00 00       	mov    $0x1,%edi
  401ced:	31 f6                	xor    %esi,%esi
  401cef:	31 d2                	xor    %edx,%edx
  401cf1:	31 c0                	xor    %eax,%eax
  401cf3:	e8 08 1c 00 00       	call   403900 <__libirc_print>
  401cf8:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401cff:	00 00 
  401d01:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  401d08:	00 
  401d09:	75 0a                	jne    401d15 <__intel_new_feature_proc_init_n+0x365>
  401d0b:	bf 01 00 00 00       	mov    $0x1,%edi
  401d10:	e8 4b f4 ff ff       	call   401160 <exit@plt>
  401d15:	e8 76 f3 ff ff       	call   401090 <__stack_chk_fail@plt>
  401d1a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401d20 <__intel_new_feature_proc_init>:
  401d20:	f3 0f 1e fa          	endbr64
  401d24:	53                   	push   %rbx
  401d25:	89 fb                	mov    %edi,%ebx
  401d27:	31 ff                	xor    %edi,%edi
  401d29:	e8 82 fc ff ff       	call   4019b0 <__intel_new_feature_proc_init_n>
  401d2e:	48 c7 c7 d0 70 40 00 	mov    $0x4070d0,%rdi
  401d35:	be 06 00 00 00       	mov    $0x6,%esi
  401d3a:	e8 61 17 00 00       	call   4034a0 <__libirc_get_cpu_feature>
  401d3f:	83 f8 01             	cmp    $0x1,%eax
  401d42:	75 0a                	jne    401d4e <__intel_new_feature_proc_init+0x2e>
  401d44:	31 ff                	xor    %edi,%edi
  401d46:	89 de                	mov    %ebx,%esi
  401d48:	5b                   	pop    %rbx
  401d49:	e9 52 00 00 00       	jmp    401da0 <__intel_proc_init_ftzdazule>
  401d4e:	85 c0                	test   %eax,%eax
  401d50:	78 02                	js     401d54 <__intel_new_feature_proc_init+0x34>
  401d52:	5b                   	pop    %rbx
  401d53:	c3                   	ret
  401d54:	bf 01 00 00 00       	mov    $0x1,%edi
  401d59:	31 f6                	xor    %esi,%esi
  401d5b:	31 d2                	xor    %edx,%edx
  401d5d:	31 c0                	xor    %eax,%eax
  401d5f:	e8 9c 1b 00 00       	call   403900 <__libirc_print>
  401d64:	bf 01 00 00 00       	mov    $0x1,%edi
  401d69:	be 3a 00 00 00       	mov    $0x3a,%esi
  401d6e:	31 d2                	xor    %edx,%edx
  401d70:	31 c0                	xor    %eax,%eax
  401d72:	e8 89 1b 00 00       	call   403900 <__libirc_print>
  401d77:	bf 01 00 00 00       	mov    $0x1,%edi
  401d7c:	31 f6                	xor    %esi,%esi
  401d7e:	31 d2                	xor    %edx,%edx
  401d80:	31 c0                	xor    %eax,%eax
  401d82:	e8 79 1b 00 00       	call   403900 <__libirc_print>
  401d87:	bf 01 00 00 00       	mov    $0x1,%edi
  401d8c:	e8 cf f3 ff ff       	call   401160 <exit@plt>
  401d91:	0f 1f 00             	nopl   (%rax)
  401d94:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401d9b:	00 00 00 
  401d9e:	66 90                	xchg   %ax,%ax

0000000000401da0 <__intel_proc_init_ftzdazule>:
  401da0:	f3 0f 1e fa          	endbr64
  401da4:	55                   	push   %rbp
  401da5:	41 56                	push   %r14
  401da7:	53                   	push   %rbx
  401da8:	48 81 ec 20 02 00 00 	sub    $0x220,%rsp
  401daf:	89 f3                	mov    %esi,%ebx
  401db1:	41 89 f6             	mov    %esi,%r14d
  401db4:	41 83 e6 04          	and    $0x4,%r14d
  401db8:	89 f5                	mov    %esi,%ebp
  401dba:	83 e5 02             	and    $0x2,%ebp
  401dbd:	74 07                	je     401dc6 <__intel_proc_init_ftzdazule+0x26>
  401dbf:	89 f8                	mov    %edi,%eax
  401dc1:	83 e0 02             	and    $0x2,%eax
  401dc4:	74 12                	je     401dd8 <__intel_proc_init_ftzdazule+0x38>
  401dc6:	31 c0                	xor    %eax,%eax
  401dc8:	45 85 f6             	test   %r14d,%r14d
  401dcb:	74 38                	je     401e05 <__intel_proc_init_ftzdazule+0x65>
  401dcd:	b8 01 00 00 00       	mov    $0x1,%eax
  401dd2:	40 f6 c7 04          	test   $0x4,%dil
  401dd6:	75 2d                	jne    401e05 <__intel_proc_init_ftzdazule+0x65>
  401dd8:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  401ddd:	ba 00 02 00 00       	mov    $0x200,%edx
  401de2:	31 f6                	xor    %esi,%esi
  401de4:	e8 c7 f2 ff ff       	call   4010b0 <memset@plt>
  401de9:	0f ae 44 24 20       	fxsave 0x20(%rsp)
  401dee:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
  401df2:	89 c1                	mov    %eax,%ecx
  401df4:	c1 e1 19             	shl    $0x19,%ecx
  401df7:	c1 f9 1f             	sar    $0x1f,%ecx
  401dfa:	21 cd                	and    %ecx,%ebp
  401dfc:	c1 e0 0e             	shl    $0xe,%eax
  401dff:	c1 f8 1f             	sar    $0x1f,%eax
  401e02:	44 21 f0             	and    %r14d,%eax
  401e05:	f6 c3 01             	test   $0x1,%bl
  401e08:	74 17                	je     401e21 <__intel_proc_init_ftzdazule+0x81>
  401e0a:	0f ae 5c 24 1c       	stmxcsr 0x1c(%rsp)
  401e0f:	b9 00 80 00 00       	mov    $0x8000,%ecx
  401e14:	0b 4c 24 1c          	or     0x1c(%rsp),%ecx
  401e18:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
  401e1c:	0f ae 54 24 18       	ldmxcsr 0x18(%rsp)
  401e21:	85 ed                	test   %ebp,%ebp
  401e23:	74 15                	je     401e3a <__intel_proc_init_ftzdazule+0x9a>
  401e25:	0f ae 5c 24 14       	stmxcsr 0x14(%rsp)
  401e2a:	8b 4c 24 14          	mov    0x14(%rsp),%ecx
  401e2e:	83 c9 40             	or     $0x40,%ecx
  401e31:	89 4c 24 10          	mov    %ecx,0x10(%rsp)
  401e35:	0f ae 54 24 10       	ldmxcsr 0x10(%rsp)
  401e3a:	85 c0                	test   %eax,%eax
  401e3c:	74 17                	je     401e55 <__intel_proc_init_ftzdazule+0xb5>
  401e3e:	0f ae 5c 24 0c       	stmxcsr 0xc(%rsp)
  401e43:	b8 00 00 02 00       	mov    $0x20000,%eax
  401e48:	0b 44 24 0c          	or     0xc(%rsp),%eax
  401e4c:	89 44 24 08          	mov    %eax,0x8(%rsp)
  401e50:	0f ae 54 24 08       	ldmxcsr 0x8(%rsp)
  401e55:	48 81 c4 20 02 00 00 	add    $0x220,%rsp
  401e5c:	5b                   	pop    %rbx
  401e5d:	41 5e                	pop    %r14
  401e5f:	5d                   	pop    %rbp
  401e60:	c3                   	ret
  401e61:	0f 1f 00             	nopl   (%rax)
  401e64:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401e6b:	00 00 00 
  401e6e:	66 90                	xchg   %ax,%ax

0000000000401e70 <__intel_cpu_features_init>:
  401e70:	f3 0f 1e fa          	endbr64
  401e74:	50                   	push   %rax
  401e75:	b8 01 00 00 00       	mov    $0x1,%eax
  401e7a:	e8 11 00 00 00       	call   401e90 <__intel_cpu_features_init_body>
  401e7f:	48 83 c4 08          	add    $0x8,%rsp
  401e83:	c3                   	ret
  401e84:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401e8b:	00 00 00 
  401e8e:	66 90                	xchg   %ax,%ax

0000000000401e90 <__intel_cpu_features_init_body>:
  401e90:	41 53                	push   %r11
  401e92:	41 52                	push   %r10
  401e94:	41 51                	push   %r9
  401e96:	41 50                	push   %r8
  401e98:	52                   	push   %rdx
  401e99:	51                   	push   %rcx
  401e9a:	56                   	push   %rsi
  401e9b:	57                   	push   %rdi
  401e9c:	55                   	push   %rbp
  401e9d:	53                   	push   %rbx
  401e9e:	48 81 ec 38 01 00 00 	sub    $0x138,%rsp
  401ea5:	44 0f 29 bc 24 20 01 	movaps %xmm15,0x120(%rsp)
  401eac:	00 00 
  401eae:	44 0f 29 b4 24 10 01 	movaps %xmm14,0x110(%rsp)
  401eb5:	00 00 
  401eb7:	44 0f 29 ac 24 00 01 	movaps %xmm13,0x100(%rsp)
  401ebe:	00 00 
  401ec0:	44 0f 29 a4 24 f0 00 	movaps %xmm12,0xf0(%rsp)
  401ec7:	00 00 
  401ec9:	44 0f 29 9c 24 e0 00 	movaps %xmm11,0xe0(%rsp)
  401ed0:	00 00 
  401ed2:	44 0f 29 94 24 d0 00 	movaps %xmm10,0xd0(%rsp)
  401ed9:	00 00 
  401edb:	44 0f 29 8c 24 c0 00 	movaps %xmm9,0xc0(%rsp)
  401ee2:	00 00 
  401ee4:	44 0f 29 84 24 b0 00 	movaps %xmm8,0xb0(%rsp)
  401eeb:	00 00 
  401eed:	0f 29 bc 24 a0 00 00 	movaps %xmm7,0xa0(%rsp)
  401ef4:	00 
  401ef5:	0f 29 b4 24 90 00 00 	movaps %xmm6,0x90(%rsp)
  401efc:	00 
  401efd:	0f 29 ac 24 80 00 00 	movaps %xmm5,0x80(%rsp)
  401f04:	00 
  401f05:	0f 29 64 24 70       	movaps %xmm4,0x70(%rsp)
  401f0a:	0f 29 5c 24 60       	movaps %xmm3,0x60(%rsp)
  401f0f:	0f 29 54 24 50       	movaps %xmm2,0x50(%rsp)
  401f14:	0f 29 4c 24 40       	movaps %xmm1,0x40(%rsp)
  401f19:	0f 29 44 24 30       	movaps %xmm0,0x30(%rsp)
  401f1e:	89 c5                	mov    %eax,%ebp
  401f20:	0f 57 c0             	xorps  %xmm0,%xmm0
  401f23:	0f 29 04 24          	movaps %xmm0,(%rsp)
  401f27:	0f 29 44 24 20       	movaps %xmm0,0x20(%rsp)
  401f2c:	48 89 e0             	mov    %rsp,%rax
  401f2f:	b9 01 00 00 00       	mov    $0x1,%ecx
  401f34:	e8 b7 15 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  401f39:	85 c0                	test   %eax,%eax
  401f3b:	0f 85 81 03 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  401f41:	31 c0                	xor    %eax,%eax
  401f43:	0f a2                	cpuid
  401f45:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  401f49:	89 5c 24 18          	mov    %ebx,0x18(%rsp)
  401f4d:	89 4c 24 14          	mov    %ecx,0x14(%rsp)
  401f51:	89 54 24 10          	mov    %edx,0x10(%rsp)
  401f55:	83 7c 24 1c 00       	cmpl   $0x0,0x1c(%rsp)
  401f5a:	0f 84 55 03 00 00    	je     4022b5 <__intel_cpu_features_init_body+0x425>
  401f60:	83 fd 01             	cmp    $0x1,%ebp
  401f63:	75 2a                	jne    401f8f <__intel_cpu_features_init_body+0xff>
  401f65:	81 7c 24 18 47 65 6e 	cmpl   $0x756e6547,0x18(%rsp)
  401f6c:	75 
  401f6d:	0f 85 42 03 00 00    	jne    4022b5 <__intel_cpu_features_init_body+0x425>
  401f73:	81 7c 24 10 69 6e 65 	cmpl   $0x49656e69,0x10(%rsp)
  401f7a:	49 
  401f7b:	0f 85 34 03 00 00    	jne    4022b5 <__intel_cpu_features_init_body+0x425>
  401f81:	81 7c 24 14 6e 74 65 	cmpl   $0x6c65746e,0x14(%rsp)
  401f88:	6c 
  401f89:	0f 85 26 03 00 00    	jne    4022b5 <__intel_cpu_features_init_body+0x425>
  401f8f:	b8 01 00 00 00       	mov    $0x1,%eax
  401f94:	0f a2                	cpuid
  401f96:	41 89 d2             	mov    %edx,%r10d
  401f99:	41 89 c8             	mov    %ecx,%r8d
  401f9c:	41 f6 c2 01          	test   $0x1,%r10b
  401fa0:	74 15                	je     401fb7 <__intel_cpu_features_init_body+0x127>
  401fa2:	48 89 e0             	mov    %rsp,%rax
  401fa5:	b9 02 00 00 00       	mov    $0x2,%ecx
  401faa:	e8 41 15 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  401faf:	85 c0                	test   %eax,%eax
  401fb1:	0f 85 0b 03 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  401fb7:	66 45 85 d2          	test   %r10w,%r10w
  401fbb:	79 15                	jns    401fd2 <__intel_cpu_features_init_body+0x142>
  401fbd:	48 89 e0             	mov    %rsp,%rax
  401fc0:	b9 03 00 00 00       	mov    $0x3,%ecx
  401fc5:	e8 26 15 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  401fca:	85 c0                	test   %eax,%eax
  401fcc:	0f 85 f0 02 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  401fd2:	41 f7 c2 00 00 80 00 	test   $0x800000,%r10d
  401fd9:	74 15                	je     401ff0 <__intel_cpu_features_init_body+0x160>
  401fdb:	48 89 e0             	mov    %rsp,%rax
  401fde:	b9 04 00 00 00       	mov    $0x4,%ecx
  401fe3:	e8 08 15 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  401fe8:	85 c0                	test   %eax,%eax
  401fea:	0f 85 d2 02 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  401ff0:	41 f7 c2 00 00 00 01 	test   $0x1000000,%r10d
  401ff7:	0f 85 54 03 00 00    	jne    402351 <__intel_cpu_features_init_body+0x4c1>
  401ffd:	41 f7 c0 00 00 00 40 	test   $0x40000000,%r8d
  402004:	74 15                	je     40201b <__intel_cpu_features_init_body+0x18b>
  402006:	48 89 e0             	mov    %rsp,%rax
  402009:	b9 12 00 00 00       	mov    $0x12,%ecx
  40200e:	e8 dd 14 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402013:	85 c0                	test   %eax,%eax
  402015:	0f 85 a7 02 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40201b:	41 f7 c2 00 00 00 01 	test   $0x1000000,%r10d
  402022:	75 10                	jne    402034 <__intel_cpu_features_init_body+0x1a4>
  402024:	b8 07 00 00 00       	mov    $0x7,%eax
  402029:	31 c9                	xor    %ecx,%ecx
  40202b:	0f a2                	cpuid
  40202d:	89 cf                	mov    %ecx,%edi
  40202f:	89 d6                	mov    %edx,%esi
  402031:	41 89 d9             	mov    %ebx,%r9d
  402034:	44 89 c8             	mov    %r9d,%eax
  402037:	f7 d0                	not    %eax
  402039:	a9 08 01 00 00       	test   $0x108,%eax
  40203e:	75 15                	jne    402055 <__intel_cpu_features_init_body+0x1c5>
  402040:	48 89 e0             	mov    %rsp,%rax
  402043:	b9 14 00 00 00       	mov    $0x14,%ecx
  402048:	e8 a3 14 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40204d:	85 c0                	test   %eax,%eax
  40204f:	0f 85 6d 02 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402055:	41 f6 c1 04          	test   $0x4,%r9b
  402059:	74 15                	je     402070 <__intel_cpu_features_init_body+0x1e0>
  40205b:	48 89 e0             	mov    %rsp,%rax
  40205e:	b9 36 00 00 00       	mov    $0x36,%ecx
  402063:	e8 88 14 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402068:	85 c0                	test   %eax,%eax
  40206a:	0f 85 52 02 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402070:	41 f6 c1 10          	test   $0x10,%r9b
  402074:	74 15                	je     40208b <__intel_cpu_features_init_body+0x1fb>
  402076:	48 89 e0             	mov    %rsp,%rax
  402079:	b9 16 00 00 00       	mov    $0x16,%ecx
  40207e:	e8 6d 14 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402083:	85 c0                	test   %eax,%eax
  402085:	0f 85 37 02 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40208b:	41 f7 c1 00 08 00 00 	test   $0x800,%r9d
  402092:	74 15                	je     4020a9 <__intel_cpu_features_init_body+0x219>
  402094:	48 89 e0             	mov    %rsp,%rax
  402097:	b9 17 00 00 00       	mov    $0x17,%ecx
  40209c:	e8 4f 14 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4020a1:	85 c0                	test   %eax,%eax
  4020a3:	0f 85 19 02 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4020a9:	41 f7 c1 00 00 08 00 	test   $0x80000,%r9d
  4020b0:	74 15                	je     4020c7 <__intel_cpu_features_init_body+0x237>
  4020b2:	48 89 e0             	mov    %rsp,%rax
  4020b5:	b9 1d 00 00 00       	mov    $0x1d,%ecx
  4020ba:	e8 31 14 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4020bf:	85 c0                	test   %eax,%eax
  4020c1:	0f 85 fb 01 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4020c7:	41 f7 c1 00 00 04 00 	test   $0x40000,%r9d
  4020ce:	74 15                	je     4020e5 <__intel_cpu_features_init_body+0x255>
  4020d0:	48 89 e0             	mov    %rsp,%rax
  4020d3:	b9 1e 00 00 00       	mov    $0x1e,%ecx
  4020d8:	e8 13 14 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4020dd:	85 c0                	test   %eax,%eax
  4020df:	0f 85 dd 01 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4020e5:	41 f7 c1 00 00 00 01 	test   $0x1000000,%r9d
  4020ec:	74 15                	je     402103 <__intel_cpu_features_init_body+0x273>
  4020ee:	48 89 e0             	mov    %rsp,%rax
  4020f1:	b9 32 00 00 00       	mov    $0x32,%ecx
  4020f6:	e8 f5 13 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4020fb:	85 c0                	test   %eax,%eax
  4020fd:	0f 85 bf 01 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402103:	b8 01 00 00 80       	mov    $0x80000001,%eax
  402108:	0f a2                	cpuid
  40210a:	f6 c1 20             	test   $0x20,%cl
  40210d:	74 15                	je     402124 <__intel_cpu_features_init_body+0x294>
  40210f:	48 89 e0             	mov    %rsp,%rax
  402112:	b9 15 00 00 00       	mov    $0x15,%ecx
  402117:	e8 d4 13 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40211c:	85 c0                	test   %eax,%eax
  40211e:	0f 85 9e 01 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402124:	b8 08 00 00 80       	mov    $0x80000008,%eax
  402129:	0f a2                	cpuid
  40212b:	f7 c3 00 02 00 00    	test   $0x200,%ebx
  402131:	74 15                	je     402148 <__intel_cpu_features_init_body+0x2b8>
  402133:	48 89 e0             	mov    %rsp,%rax
  402136:	b9 37 00 00 00       	mov    $0x37,%ecx
  40213b:	e8 b0 13 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402140:	85 c0                	test   %eax,%eax
  402142:	0f 85 7a 01 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402148:	40 f6 c7 20          	test   $0x20,%dil
  40214c:	74 15                	je     402163 <__intel_cpu_features_init_body+0x2d3>
  40214e:	48 89 e0             	mov    %rsp,%rax
  402151:	b9 3e 00 00 00       	mov    $0x3e,%ecx
  402156:	e8 95 13 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40215b:	85 c0                	test   %eax,%eax
  40215d:	0f 85 5f 01 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402163:	40 84 ff             	test   %dil,%dil
  402166:	79 15                	jns    40217d <__intel_cpu_features_init_body+0x2ed>
  402168:	48 89 e0             	mov    %rsp,%rax
  40216b:	b9 35 00 00 00       	mov    $0x35,%ecx
  402170:	e8 7b 13 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402175:	85 c0                	test   %eax,%eax
  402177:	0f 85 45 01 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40217d:	f7 c7 00 01 00 00    	test   $0x100,%edi
  402183:	74 15                	je     40219a <__intel_cpu_features_init_body+0x30a>
  402185:	48 89 e0             	mov    %rsp,%rax
  402188:	b9 2e 00 00 00       	mov    $0x2e,%ecx
  40218d:	e8 5e 13 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402192:	85 c0                	test   %eax,%eax
  402194:	0f 85 28 01 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40219a:	f7 c7 00 00 40 00    	test   $0x400000,%edi
  4021a0:	74 15                	je     4021b7 <__intel_cpu_features_init_body+0x327>
  4021a2:	48 89 e0             	mov    %rsp,%rax
  4021a5:	b9 33 00 00 00       	mov    $0x33,%ecx
  4021aa:	e8 41 13 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4021af:	85 c0                	test   %eax,%eax
  4021b1:	0f 85 0b 01 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4021b7:	f7 c7 00 00 00 01    	test   $0x1000000,%edi
  4021bd:	74 15                	je     4021d4 <__intel_cpu_features_init_body+0x344>
  4021bf:	48 89 e0             	mov    %rsp,%rax
  4021c2:	b9 3b 00 00 00       	mov    $0x3b,%ecx
  4021c7:	e8 24 13 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4021cc:	85 c0                	test   %eax,%eax
  4021ce:	0f 85 ee 00 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4021d4:	f7 c7 00 00 00 08    	test   $0x8000000,%edi
  4021da:	74 15                	je     4021f1 <__intel_cpu_features_init_body+0x361>
  4021dc:	48 89 e0             	mov    %rsp,%rax
  4021df:	b9 3c 00 00 00       	mov    $0x3c,%ecx
  4021e4:	e8 07 13 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4021e9:	85 c0                	test   %eax,%eax
  4021eb:	0f 85 d1 00 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4021f1:	f7 c7 00 00 00 10    	test   $0x10000000,%edi
  4021f7:	74 15                	je     40220e <__intel_cpu_features_init_body+0x37e>
  4021f9:	48 89 e0             	mov    %rsp,%rax
  4021fc:	b9 3d 00 00 00       	mov    $0x3d,%ecx
  402201:	e8 ea 12 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402206:	85 c0                	test   %eax,%eax
  402208:	0f 85 b4 00 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40220e:	f7 c7 00 00 00 20    	test   $0x20000000,%edi
  402214:	74 15                	je     40222b <__intel_cpu_features_init_body+0x39b>
  402216:	48 89 e0             	mov    %rsp,%rax
  402219:	b9 40 00 00 00       	mov    $0x40,%ecx
  40221e:	e8 cd 12 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402223:	85 c0                	test   %eax,%eax
  402225:	0f 85 97 00 00 00    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40222b:	f7 c6 00 00 10 00    	test   $0x100000,%esi
  402231:	74 11                	je     402244 <__intel_cpu_features_init_body+0x3b4>
  402233:	48 89 e0             	mov    %rsp,%rax
  402236:	b9 34 00 00 00       	mov    $0x34,%ecx
  40223b:	e8 b0 12 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402240:	85 c0                	test   %eax,%eax
  402242:	75 7e                	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402244:	f7 c6 00 00 04 00    	test   $0x40000,%esi
  40224a:	74 11                	je     40225d <__intel_cpu_features_init_body+0x3cd>
  40224c:	48 89 e0             	mov    %rsp,%rax
  40224f:	b9 38 00 00 00       	mov    $0x38,%ecx
  402254:	e8 97 12 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402259:	85 c0                	test   %eax,%eax
  40225b:	75 65                	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40225d:	b8 14 00 00 00       	mov    $0x14,%eax
  402262:	31 c9                	xor    %ecx,%ecx
  402264:	0f a2                	cpuid
  402266:	f6 c3 10             	test   $0x10,%bl
  402269:	74 11                	je     40227c <__intel_cpu_features_init_body+0x3ec>
  40226b:	48 89 e0             	mov    %rsp,%rax
  40226e:	b9 1b 00 00 00       	mov    $0x1b,%ecx
  402273:	e8 78 12 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402278:	85 c0                	test   %eax,%eax
  40227a:	75 46                	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40227c:	f7 c7 00 00 80 00    	test   $0x800000,%edi
  402282:	0f 85 3a 02 00 00    	jne    4024c2 <__intel_cpu_features_init_body+0x632>
  402288:	41 f7 c0 00 00 00 08 	test   $0x8000000,%r8d
  40228f:	0f 85 88 02 00 00    	jne    40251d <__intel_cpu_features_init_body+0x68d>
  402295:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  40229a:	e8 b1 12 00 00       	call   403550 <__libirc_handle_intel_isa_disable>
  40229f:	85 c0                	test   %eax,%eax
  4022a1:	0f 8e 09 06 00 00    	jle    4028b0 <__intel_cpu_features_init_body+0xa20>
  4022a7:	0f 28 44 24 20       	movaps 0x20(%rsp),%xmm0
  4022ac:	0f 55 04 24          	andnps (%rsp),%xmm0
  4022b0:	e9 ff 05 00 00       	jmp    4028b4 <__intel_cpu_features_init_body+0xa24>
  4022b5:	0f 28 04 24          	movaps (%rsp),%xmm0
  4022b9:	0f 29 05 10 4e 00 00 	movaps %xmm0,0x4e10(%rip)        # 4070d0 <__intel_cpu_feature_indicator>
  4022c0:	31 c0                	xor    %eax,%eax
  4022c2:	0f 28 44 24 30       	movaps 0x30(%rsp),%xmm0
  4022c7:	0f 28 4c 24 40       	movaps 0x40(%rsp),%xmm1
  4022cc:	0f 28 54 24 50       	movaps 0x50(%rsp),%xmm2
  4022d1:	0f 28 5c 24 60       	movaps 0x60(%rsp),%xmm3
  4022d6:	0f 28 64 24 70       	movaps 0x70(%rsp),%xmm4
  4022db:	0f 28 ac 24 80 00 00 	movaps 0x80(%rsp),%xmm5
  4022e2:	00 
  4022e3:	0f 28 b4 24 90 00 00 	movaps 0x90(%rsp),%xmm6
  4022ea:	00 
  4022eb:	0f 28 bc 24 a0 00 00 	movaps 0xa0(%rsp),%xmm7
  4022f2:	00 
  4022f3:	44 0f 28 84 24 b0 00 	movaps 0xb0(%rsp),%xmm8
  4022fa:	00 00 
  4022fc:	44 0f 28 8c 24 c0 00 	movaps 0xc0(%rsp),%xmm9
  402303:	00 00 
  402305:	44 0f 28 94 24 d0 00 	movaps 0xd0(%rsp),%xmm10
  40230c:	00 00 
  40230e:	44 0f 28 9c 24 e0 00 	movaps 0xe0(%rsp),%xmm11
  402315:	00 00 
  402317:	44 0f 28 a4 24 f0 00 	movaps 0xf0(%rsp),%xmm12
  40231e:	00 00 
  402320:	44 0f 28 ac 24 00 01 	movaps 0x100(%rsp),%xmm13
  402327:	00 00 
  402329:	44 0f 28 b4 24 10 01 	movaps 0x110(%rsp),%xmm14
  402330:	00 00 
  402332:	44 0f 28 bc 24 20 01 	movaps 0x120(%rsp),%xmm15
  402339:	00 00 
  40233b:	48 81 c4 38 01 00 00 	add    $0x138,%rsp
  402342:	5b                   	pop    %rbx
  402343:	5d                   	pop    %rbp
  402344:	5f                   	pop    %rdi
  402345:	5e                   	pop    %rsi
  402346:	59                   	pop    %rcx
  402347:	5a                   	pop    %rdx
  402348:	41 58                	pop    %r8
  40234a:	41 59                	pop    %r9
  40234c:	41 5a                	pop    %r10
  40234e:	41 5b                	pop    %r11
  402350:	c3                   	ret
  402351:	48 89 e0             	mov    %rsp,%rax
  402354:	b9 05 00 00 00       	mov    $0x5,%ecx
  402359:	e8 92 11 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40235e:	85 c0                	test   %eax,%eax
  402360:	0f 85 5c ff ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402366:	41 f7 c2 00 00 00 02 	test   $0x2000000,%r10d
  40236d:	74 15                	je     402384 <__intel_cpu_features_init_body+0x4f4>
  40236f:	48 89 e0             	mov    %rsp,%rax
  402372:	b9 06 00 00 00       	mov    $0x6,%ecx
  402377:	e8 74 11 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40237c:	85 c0                	test   %eax,%eax
  40237e:	0f 85 3e ff ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402384:	41 f7 c2 00 00 00 04 	test   $0x4000000,%r10d
  40238b:	74 15                	je     4023a2 <__intel_cpu_features_init_body+0x512>
  40238d:	48 89 e0             	mov    %rsp,%rax
  402390:	b9 07 00 00 00       	mov    $0x7,%ecx
  402395:	e8 56 11 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40239a:	85 c0                	test   %eax,%eax
  40239c:	0f 85 20 ff ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4023a2:	41 f6 c0 01          	test   $0x1,%r8b
  4023a6:	74 15                	je     4023bd <__intel_cpu_features_init_body+0x52d>
  4023a8:	48 89 e0             	mov    %rsp,%rax
  4023ab:	b9 08 00 00 00       	mov    $0x8,%ecx
  4023b0:	e8 3b 11 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4023b5:	85 c0                	test   %eax,%eax
  4023b7:	0f 85 05 ff ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4023bd:	41 f7 c0 00 02 00 00 	test   $0x200,%r8d
  4023c4:	74 15                	je     4023db <__intel_cpu_features_init_body+0x54b>
  4023c6:	48 89 e0             	mov    %rsp,%rax
  4023c9:	b9 09 00 00 00       	mov    $0x9,%ecx
  4023ce:	e8 1d 11 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4023d3:	85 c0                	test   %eax,%eax
  4023d5:	0f 85 e7 fe ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4023db:	41 f7 c0 00 00 40 00 	test   $0x400000,%r8d
  4023e2:	74 15                	je     4023f9 <__intel_cpu_features_init_body+0x569>
  4023e4:	48 89 e0             	mov    %rsp,%rax
  4023e7:	b9 0c 00 00 00       	mov    $0xc,%ecx
  4023ec:	e8 ff 10 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4023f1:	85 c0                	test   %eax,%eax
  4023f3:	0f 85 c9 fe ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4023f9:	41 f7 c0 00 00 08 00 	test   $0x80000,%r8d
  402400:	74 15                	je     402417 <__intel_cpu_features_init_body+0x587>
  402402:	48 89 e0             	mov    %rsp,%rax
  402405:	b9 0a 00 00 00       	mov    $0xa,%ecx
  40240a:	e8 e1 10 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40240f:	85 c0                	test   %eax,%eax
  402411:	0f 85 ab fe ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402417:	41 f7 c0 00 00 10 00 	test   $0x100000,%r8d
  40241e:	74 15                	je     402435 <__intel_cpu_features_init_body+0x5a5>
  402420:	48 89 e0             	mov    %rsp,%rax
  402423:	b9 0b 00 00 00       	mov    $0xb,%ecx
  402428:	e8 c3 10 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40242d:	85 c0                	test   %eax,%eax
  40242f:	0f 85 8d fe ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402435:	41 f7 c0 00 00 80 00 	test   $0x800000,%r8d
  40243c:	74 15                	je     402453 <__intel_cpu_features_init_body+0x5c3>
  40243e:	48 89 e0             	mov    %rsp,%rax
  402441:	b9 0d 00 00 00       	mov    $0xd,%ecx
  402446:	e8 a5 10 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40244b:	85 c0                	test   %eax,%eax
  40244d:	0f 85 6f fe ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402453:	41 f6 c0 02          	test   $0x2,%r8b
  402457:	74 15                	je     40246e <__intel_cpu_features_init_body+0x5de>
  402459:	48 89 e0             	mov    %rsp,%rax
  40245c:	b9 0e 00 00 00       	mov    $0xe,%ecx
  402461:	e8 8a 10 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402466:	85 c0                	test   %eax,%eax
  402468:	0f 85 54 fe ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40246e:	41 f7 c0 00 00 00 02 	test   $0x2000000,%r8d
  402475:	74 15                	je     40248c <__intel_cpu_features_init_body+0x5fc>
  402477:	48 89 e0             	mov    %rsp,%rax
  40247a:	b9 0f 00 00 00       	mov    $0xf,%ecx
  40247f:	e8 6c 10 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402484:	85 c0                	test   %eax,%eax
  402486:	0f 85 36 fe ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40248c:	b8 07 00 00 00       	mov    $0x7,%eax
  402491:	31 c9                	xor    %ecx,%ecx
  402493:	0f a2                	cpuid
  402495:	89 cf                	mov    %ecx,%edi
  402497:	89 d6                	mov    %edx,%esi
  402499:	41 89 d9             	mov    %ebx,%r9d
  40249c:	f7 c3 00 00 00 20    	test   $0x20000000,%ebx
  4024a2:	0f 84 55 fb ff ff    	je     401ffd <__intel_cpu_features_init_body+0x16d>
  4024a8:	48 89 e0             	mov    %rsp,%rax
  4024ab:	b9 24 00 00 00       	mov    $0x24,%ecx
  4024b0:	e8 3b 10 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4024b5:	85 c0                	test   %eax,%eax
  4024b7:	0f 85 05 fe ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4024bd:	e9 3b fb ff ff       	jmp    401ffd <__intel_cpu_features_init_body+0x16d>
  4024c2:	48 89 e0             	mov    %rsp,%rax
  4024c5:	b9 01 00 00 00       	mov    $0x1,%ecx
  4024ca:	e8 21 10 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4024cf:	85 c0                	test   %eax,%eax
  4024d1:	0f 85 eb fd ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4024d7:	b8 19 00 00 00       	mov    $0x19,%eax
  4024dc:	31 c9                	xor    %ecx,%ecx
  4024de:	0f a2                	cpuid
  4024e0:	f6 c3 01             	test   $0x1,%bl
  4024e3:	74 15                	je     4024fa <__intel_cpu_features_init_body+0x66a>
  4024e5:	48 89 e0             	mov    %rsp,%rax
  4024e8:	b9 45 00 00 00       	mov    $0x45,%ecx
  4024ed:	e8 fe 0f 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4024f2:	85 c0                	test   %eax,%eax
  4024f4:	0f 85 c8 fd ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4024fa:	f6 c3 04             	test   $0x4,%bl
  4024fd:	0f 84 85 fd ff ff    	je     402288 <__intel_cpu_features_init_body+0x3f8>
  402503:	48 89 e0             	mov    %rsp,%rax
  402506:	b9 46 00 00 00       	mov    $0x46,%ecx
  40250b:	e8 e0 0f 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402510:	85 c0                	test   %eax,%eax
  402512:	0f 85 aa fd ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402518:	e9 6b fd ff ff       	jmp    402288 <__intel_cpu_features_init_body+0x3f8>
  40251d:	48 89 e0             	mov    %rsp,%rax
  402520:	b9 01 00 00 00       	mov    $0x1,%ecx
  402525:	e8 c6 0f 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40252a:	85 c0                	test   %eax,%eax
  40252c:	0f 85 90 fd ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402532:	31 c9                	xor    %ecx,%ecx
  402534:	0f 01 d0             	xgetbv
  402537:	41 89 c2             	mov    %eax,%r10d
  40253a:	41 f7 d2             	not    %r10d
  40253d:	41 f7 c2 00 00 06 00 	test   $0x60000,%r10d
  402544:	75 6c                	jne    4025b2 <__intel_cpu_features_init_body+0x722>
  402546:	48 89 e0             	mov    %rsp,%rax
  402549:	b9 01 00 00 00       	mov    $0x1,%ecx
  40254e:	e8 9d 0f 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402553:	85 c0                	test   %eax,%eax
  402555:	0f 85 67 fd ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40255b:	f7 c6 00 00 00 01    	test   $0x1000000,%esi
  402561:	74 15                	je     402578 <__intel_cpu_features_init_body+0x6e8>
  402563:	48 89 e0             	mov    %rsp,%rax
  402566:	b9 42 00 00 00       	mov    $0x42,%ecx
  40256b:	e8 80 0f 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402570:	85 c0                	test   %eax,%eax
  402572:	0f 85 4a fd ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402578:	f7 c6 00 00 00 02    	test   $0x2000000,%esi
  40257e:	74 15                	je     402595 <__intel_cpu_features_init_body+0x705>
  402580:	48 89 e0             	mov    %rsp,%rax
  402583:	b9 43 00 00 00       	mov    $0x43,%ecx
  402588:	e8 63 0f 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40258d:	85 c0                	test   %eax,%eax
  40258f:	0f 85 2d fd ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402595:	f7 c6 00 00 40 00    	test   $0x400000,%esi
  40259b:	74 15                	je     4025b2 <__intel_cpu_features_init_body+0x722>
  40259d:	48 89 e0             	mov    %rsp,%rax
  4025a0:	b9 44 00 00 00       	mov    $0x44,%ecx
  4025a5:	e8 46 0f 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4025aa:	85 c0                	test   %eax,%eax
  4025ac:	0f 85 10 fd ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4025b2:	41 f6 c2 06          	test   $0x6,%r10b
  4025b6:	0f 85 d9 fc ff ff    	jne    402295 <__intel_cpu_features_init_body+0x405>
  4025bc:	48 89 e0             	mov    %rsp,%rax
  4025bf:	b9 01 00 00 00       	mov    $0x1,%ecx
  4025c4:	e8 27 0f 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4025c9:	85 c0                	test   %eax,%eax
  4025cb:	0f 85 f1 fc ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4025d1:	41 f7 c0 00 00 00 10 	test   $0x10000000,%r8d
  4025d8:	0f 85 f1 02 00 00    	jne    4028cf <__intel_cpu_features_init_body+0xa3f>
  4025de:	41 f7 c0 00 00 00 20 	test   $0x20000000,%r8d
  4025e5:	74 15                	je     4025fc <__intel_cpu_features_init_body+0x76c>
  4025e7:	48 89 e0             	mov    %rsp,%rax
  4025ea:	b9 11 00 00 00       	mov    $0x11,%ecx
  4025ef:	e8 fc 0e 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4025f4:	85 c0                	test   %eax,%eax
  4025f6:	0f 85 c6 fc ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4025fc:	41 f6 c1 20          	test   $0x20,%r9b
  402600:	74 15                	je     402617 <__intel_cpu_features_init_body+0x787>
  402602:	48 89 e0             	mov    %rsp,%rax
  402605:	b9 18 00 00 00       	mov    $0x18,%ecx
  40260a:	e8 e1 0e 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40260f:	85 c0                	test   %eax,%eax
  402611:	0f 85 ab fc ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402617:	41 f7 c0 00 10 00 00 	test   $0x1000,%r8d
  40261e:	74 15                	je     402635 <__intel_cpu_features_init_body+0x7a5>
  402620:	48 89 e0             	mov    %rsp,%rax
  402623:	b9 13 00 00 00       	mov    $0x13,%ecx
  402628:	e8 c3 0e 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40262d:	85 c0                	test   %eax,%eax
  40262f:	0f 85 8d fc ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402635:	41 f6 c2 18          	test   $0x18,%r10b
  402639:	75 33                	jne    40266e <__intel_cpu_features_init_body+0x7de>
  40263b:	48 89 e0             	mov    %rsp,%rax
  40263e:	b9 01 00 00 00       	mov    $0x1,%ecx
  402643:	e8 a8 0e 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402648:	85 c0                	test   %eax,%eax
  40264a:	0f 85 72 fc ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402650:	41 f7 c1 00 40 00 00 	test   $0x4000,%r9d
  402657:	74 15                	je     40266e <__intel_cpu_features_init_body+0x7de>
  402659:	48 89 e0             	mov    %rsp,%rax
  40265c:	b9 25 00 00 00       	mov    $0x25,%ecx
  402661:	e8 8a 0e 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402666:	85 c0                	test   %eax,%eax
  402668:	0f 85 54 fc ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40266e:	b8 07 00 00 00       	mov    $0x7,%eax
  402673:	b9 01 00 00 00       	mov    $0x1,%ecx
  402678:	0f a2                	cpuid
  40267a:	89 c2                	mov    %eax,%edx
  40267c:	f6 c2 10             	test   $0x10,%dl
  40267f:	74 15                	je     402696 <__intel_cpu_features_init_body+0x806>
  402681:	48 89 e0             	mov    %rsp,%rax
  402684:	b9 41 00 00 00       	mov    $0x41,%ecx
  402689:	e8 62 0e 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40268e:	85 c0                	test   %eax,%eax
  402690:	0f 85 2c fc ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402696:	41 f6 c2 e0          	test   $0xe0,%r10b
  40269a:	0f 85 f5 fb ff ff    	jne    402295 <__intel_cpu_features_init_body+0x405>
  4026a0:	48 89 e0             	mov    %rsp,%rax
  4026a3:	b9 01 00 00 00       	mov    $0x1,%ecx
  4026a8:	e8 43 0e 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4026ad:	85 c0                	test   %eax,%eax
  4026af:	0f 85 0d fc ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4026b5:	41 f7 c1 00 00 01 00 	test   $0x10000,%r9d
  4026bc:	74 15                	je     4026d3 <__intel_cpu_features_init_body+0x843>
  4026be:	48 89 e0             	mov    %rsp,%rax
  4026c1:	b9 19 00 00 00       	mov    $0x19,%ecx
  4026c6:	e8 25 0e 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4026cb:	85 c0                	test   %eax,%eax
  4026cd:	0f 85 ef fb ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4026d3:	41 f7 c1 00 00 00 10 	test   $0x10000000,%r9d
  4026da:	74 15                	je     4026f1 <__intel_cpu_features_init_body+0x861>
  4026dc:	48 89 e0             	mov    %rsp,%rax
  4026df:	b9 23 00 00 00       	mov    $0x23,%ecx
  4026e4:	e8 07 0e 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4026e9:	85 c0                	test   %eax,%eax
  4026eb:	0f 85 d1 fb ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4026f1:	41 f7 c1 00 00 00 08 	test   $0x8000000,%r9d
  4026f8:	74 15                	je     40270f <__intel_cpu_features_init_body+0x87f>
  4026fa:	48 89 e0             	mov    %rsp,%rax
  4026fd:	b9 21 00 00 00       	mov    $0x21,%ecx
  402702:	e8 e9 0d 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402707:	85 c0                	test   %eax,%eax
  402709:	0f 85 b3 fb ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40270f:	41 f7 c1 00 00 00 04 	test   $0x4000000,%r9d
  402716:	74 15                	je     40272d <__intel_cpu_features_init_body+0x89d>
  402718:	48 89 e0             	mov    %rsp,%rax
  40271b:	b9 22 00 00 00       	mov    $0x22,%ecx
  402720:	e8 cb 0d 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402725:	85 c0                	test   %eax,%eax
  402727:	0f 85 95 fb ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40272d:	41 f7 c1 00 00 02 00 	test   $0x20000,%r9d
  402734:	74 15                	je     40274b <__intel_cpu_features_init_body+0x8bb>
  402736:	48 89 e0             	mov    %rsp,%rax
  402739:	b9 1a 00 00 00       	mov    $0x1a,%ecx
  40273e:	e8 ad 0d 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402743:	85 c0                	test   %eax,%eax
  402745:	0f 85 77 fb ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40274b:	41 f7 c1 00 00 00 40 	test   $0x40000000,%r9d
  402752:	74 15                	je     402769 <__intel_cpu_features_init_body+0x8d9>
  402754:	48 89 e0             	mov    %rsp,%rax
  402757:	b9 26 00 00 00       	mov    $0x26,%ecx
  40275c:	e8 8f 0d 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402761:	85 c0                	test   %eax,%eax
  402763:	0f 85 59 fb ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402769:	45 85 c9             	test   %r9d,%r9d
  40276c:	0f 88 b5 01 00 00    	js     402927 <__intel_cpu_features_init_body+0xa97>
  402772:	41 f7 c1 00 00 20 00 	test   $0x200000,%r9d
  402779:	74 15                	je     402790 <__intel_cpu_features_init_body+0x900>
  40277b:	48 89 e0             	mov    %rsp,%rax
  40277e:	b9 1f 00 00 00       	mov    $0x1f,%ecx
  402783:	e8 68 0d 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402788:	85 c0                	test   %eax,%eax
  40278a:	0f 85 32 fb ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402790:	40 f6 c7 02          	test   $0x2,%dil
  402794:	74 15                	je     4027ab <__intel_cpu_features_init_body+0x91b>
  402796:	48 89 e0             	mov    %rsp,%rax
  402799:	b9 28 00 00 00       	mov    $0x28,%ecx
  40279e:	e8 4d 0d 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4027a3:	85 c0                	test   %eax,%eax
  4027a5:	0f 85 17 fb ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4027ab:	f7 c7 00 40 00 00    	test   $0x4000,%edi
  4027b1:	74 15                	je     4027c8 <__intel_cpu_features_init_body+0x938>
  4027b3:	48 89 e0             	mov    %rsp,%rax
  4027b6:	b9 2b 00 00 00       	mov    $0x2b,%ecx
  4027bb:	e8 30 0d 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4027c0:	85 c0                	test   %eax,%eax
  4027c2:	0f 85 fa fa ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4027c8:	40 f6 c6 04          	test   $0x4,%sil
  4027cc:	74 15                	je     4027e3 <__intel_cpu_features_init_body+0x953>
  4027ce:	48 89 e0             	mov    %rsp,%rax
  4027d1:	b9 2a 00 00 00       	mov    $0x2a,%ecx
  4027d6:	e8 15 0d 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4027db:	85 c0                	test   %eax,%eax
  4027dd:	0f 85 df fa ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4027e3:	40 f6 c6 08          	test   $0x8,%sil
  4027e7:	74 15                	je     4027fe <__intel_cpu_features_init_body+0x96e>
  4027e9:	48 89 e0             	mov    %rsp,%rax
  4027ec:	b9 29 00 00 00       	mov    $0x29,%ecx
  4027f1:	e8 fa 0c 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4027f6:	85 c0                	test   %eax,%eax
  4027f8:	0f 85 c4 fa ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4027fe:	f7 c7 00 10 00 00    	test   $0x1000,%edi
  402804:	74 15                	je     40281b <__intel_cpu_features_init_body+0x98b>
  402806:	48 89 e0             	mov    %rsp,%rax
  402809:	b9 2c 00 00 00       	mov    $0x2c,%ecx
  40280e:	e8 dd 0c 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402813:	85 c0                	test   %eax,%eax
  402815:	0f 85 a7 fa ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40281b:	40 f6 c7 40          	test   $0x40,%dil
  40281f:	74 15                	je     402836 <__intel_cpu_features_init_body+0x9a6>
  402821:	48 89 e0             	mov    %rsp,%rax
  402824:	b9 2d 00 00 00       	mov    $0x2d,%ecx
  402829:	e8 c2 0c 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40282e:	85 c0                	test   %eax,%eax
  402830:	0f 85 8c fa ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402836:	f7 c7 00 08 00 00    	test   $0x800,%edi
  40283c:	74 15                	je     402853 <__intel_cpu_features_init_body+0x9c3>
  40283e:	48 89 e0             	mov    %rsp,%rax
  402841:	b9 31 00 00 00       	mov    $0x31,%ecx
  402846:	e8 a5 0c 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40284b:	85 c0                	test   %eax,%eax
  40284d:	0f 85 6f fa ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402853:	f6 c2 20             	test   $0x20,%dl
  402856:	74 15                	je     40286d <__intel_cpu_features_init_body+0x9dd>
  402858:	48 89 e0             	mov    %rsp,%rax
  40285b:	b9 3f 00 00 00       	mov    $0x3f,%ecx
  402860:	e8 8b 0c 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402865:	85 c0                	test   %eax,%eax
  402867:	0f 85 55 fa ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40286d:	f7 c6 00 00 80 00    	test   $0x800000,%esi
  402873:	74 15                	je     40288a <__intel_cpu_features_init_body+0x9fa>
  402875:	48 89 e0             	mov    %rsp,%rax
  402878:	b9 3a 00 00 00       	mov    $0x3a,%ecx
  40287d:	e8 6e 0c 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402882:	85 c0                	test   %eax,%eax
  402884:	0f 85 38 fa ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40288a:	f7 c6 00 01 00 00    	test   $0x100,%esi
  402890:	0f 84 ff f9 ff ff    	je     402295 <__intel_cpu_features_init_body+0x405>
  402896:	48 89 e0             	mov    %rsp,%rax
  402899:	b9 39 00 00 00       	mov    $0x39,%ecx
  40289e:	e8 4d 0c 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4028a3:	85 c0                	test   %eax,%eax
  4028a5:	0f 85 17 fa ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4028ab:	e9 e5 f9 ff ff       	jmp    402295 <__intel_cpu_features_init_body+0x405>
  4028b0:	0f 28 04 24          	movaps (%rsp),%xmm0
  4028b4:	83 fd 01             	cmp    $0x1,%ebp
  4028b7:	75 07                	jne    4028c0 <__intel_cpu_features_init_body+0xa30>
  4028b9:	0f 29 05 10 48 00 00 	movaps %xmm0,0x4810(%rip)        # 4070d0 <__intel_cpu_feature_indicator>
  4028c0:	48 c7 c0 e0 70 40 00 	mov    $0x4070e0,%rax
  4028c7:	0f 29 00             	movaps %xmm0,(%rax)
  4028ca:	e9 f1 f9 ff ff       	jmp    4022c0 <__intel_cpu_features_init_body+0x430>
  4028cf:	48 89 e0             	mov    %rsp,%rax
  4028d2:	b9 10 00 00 00       	mov    $0x10,%ecx
  4028d7:	e8 14 0c 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4028dc:	85 c0                	test   %eax,%eax
  4028de:	0f 85 de f9 ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  4028e4:	f7 c7 00 02 00 00    	test   $0x200,%edi
  4028ea:	74 15                	je     402901 <__intel_cpu_features_init_body+0xa71>
  4028ec:	48 89 e0             	mov    %rsp,%rax
  4028ef:	b9 2f 00 00 00       	mov    $0x2f,%ecx
  4028f4:	e8 f7 0b 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  4028f9:	85 c0                	test   %eax,%eax
  4028fb:	0f 85 c1 f9 ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402901:	f7 c7 00 04 00 00    	test   $0x400,%edi
  402907:	0f 84 d1 fc ff ff    	je     4025de <__intel_cpu_features_init_body+0x74e>
  40290d:	48 89 e0             	mov    %rsp,%rax
  402910:	b9 30 00 00 00       	mov    $0x30,%ecx
  402915:	e8 d6 0b 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  40291a:	85 c0                	test   %eax,%eax
  40291c:	0f 85 a0 f9 ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  402922:	e9 b7 fc ff ff       	jmp    4025de <__intel_cpu_features_init_body+0x74e>
  402927:	48 89 e0             	mov    %rsp,%rax
  40292a:	b9 27 00 00 00       	mov    $0x27,%ecx
  40292f:	e8 bc 0b 00 00       	call   4034f0 <__libirc_set_cpu_feature>
  402934:	85 c0                	test   %eax,%eax
  402936:	0f 85 86 f9 ff ff    	jne    4022c2 <__intel_cpu_features_init_body+0x432>
  40293c:	e9 31 fe ff ff       	jmp    402772 <__intel_cpu_features_init_body+0x8e2>
  402941:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  402948:	00 00 00 
  40294b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000402950 <__intel_cpu_features_init_x>:
  402950:	f3 0f 1e fa          	endbr64
  402954:	50                   	push   %rax
  402955:	31 c0                	xor    %eax,%eax
  402957:	e8 34 f5 ff ff       	call   401e90 <__intel_cpu_features_init_body>
  40295c:	48 83 c4 08          	add    $0x8,%rsp
  402960:	c3                   	ret
  402961:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  402968:	00 00 00 
  40296b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000402970 <__libirc_get_feature_name>:
  402970:	f3 0f 1e fa          	endbr64
  402974:	50                   	push   %rax
  402975:	80 3d 74 47 00 00 00 	cmpb   $0x0,0x4774(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  40297c:	75 05                	jne    402983 <__libirc_get_feature_name+0x13>
  40297e:	e8 1d 00 00 00       	call   4029a0 <__libirc_isa_init_once>
  402983:	89 f8                	mov    %edi,%eax
  402985:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  402989:	48 8d 0d 70 47 00 00 	lea    0x4770(%rip),%rcx        # 407100 <proc_info_features>
  402990:	48 8b 04 c1          	mov    (%rcx,%rax,8),%rax
  402994:	59                   	pop    %rcx
  402995:	c3                   	ret
  402996:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  40299d:	00 00 00 

00000000004029a0 <__libirc_isa_init_once>:
  4029a0:	51                   	push   %rcx
  4029a1:	80 3d 48 47 00 00 00 	cmpb   $0x0,0x4748(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  4029a8:	0f 85 aa 0a 00 00    	jne    403458 <__libirc_isa_init_once+0xab8>
  4029ae:	b8 c8 00 00 00       	mov    $0xc8,%eax
  4029b3:	48 8d 0d 46 47 00 00 	lea    0x4746(%rip),%rcx        # 407100 <proc_info_features>
  4029ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4029c0:	c7 84 08 58 ff ff ff 	movl   $0xffffffff,-0xa8(%rax,%rcx,1)
  4029c7:	ff ff ff ff 
  4029cb:	c7 84 08 70 ff ff ff 	movl   $0xffffffff,-0x90(%rax,%rcx,1)
  4029d2:	ff ff ff ff 
  4029d6:	c7 44 08 88 ff ff ff 	movl   $0xffffffff,-0x78(%rax,%rcx,1)
  4029dd:	ff 
  4029de:	c7 44 08 a0 ff ff ff 	movl   $0xffffffff,-0x60(%rax,%rcx,1)
  4029e5:	ff 
  4029e6:	c7 44 08 b8 ff ff ff 	movl   $0xffffffff,-0x48(%rax,%rcx,1)
  4029ed:	ff 
  4029ee:	c7 44 08 d0 ff ff ff 	movl   $0xffffffff,-0x30(%rax,%rcx,1)
  4029f5:	ff 
  4029f6:	c7 44 08 e8 ff ff ff 	movl   $0xffffffff,-0x18(%rax,%rcx,1)
  4029fd:	ff 
  4029fe:	c7 04 08 ff ff ff ff 	movl   $0xffffffff,(%rax,%rcx,1)
  402a05:	48 05 c0 00 00 00    	add    $0xc0,%rax
  402a0b:	48 3d c8 06 00 00    	cmp    $0x6c8,%rax
  402a11:	75 ad                	jne    4029c0 <__libirc_isa_init_once+0x20>
  402a13:	c7 05 03 4d 00 00 ff 	movl   $0xffffffff,0x4d03(%rip)        # 407720 <proc_info_features+0x620>
  402a1a:	ff ff ff 
  402a1d:	c7 05 11 4d 00 00 ff 	movl   $0xffffffff,0x4d11(%rip)        # 407738 <proc_info_features+0x638>
  402a24:	ff ff ff 
  402a27:	c7 05 1f 4d 00 00 ff 	movl   $0xffffffff,0x4d1f(%rip)        # 407750 <proc_info_features+0x650>
  402a2e:	ff ff ff 
  402a31:	c7 05 2d 4d 00 00 ff 	movl   $0xffffffff,0x4d2d(%rip)        # 407768 <proc_info_features+0x668>
  402a38:	ff ff ff 
  402a3b:	c7 05 3b 4d 00 00 ff 	movl   $0xffffffff,0x4d3b(%rip)        # 407780 <proc_info_features+0x680>
  402a42:	ff ff ff 
  402a45:	c7 05 49 4d 00 00 ff 	movl   $0xffffffff,0x4d49(%rip)        # 407798 <proc_info_features+0x698>
  402a4c:	ff ff ff 
  402a4f:	48 8d 05 53 16 00 00 	lea    0x1653(%rip),%rax        # 4040a9 <_IO_stdin_used+0xa9>
  402a56:	48 89 05 bb 46 00 00 	mov    %rax,0x46bb(%rip)        # 407118 <proc_info_features+0x18>
  402a5d:	c7 05 b9 46 00 00 00 	movl   $0x0,0x46b9(%rip)        # 407120 <proc_info_features+0x20>
  402a64:	00 00 00 
  402a67:	48 8d 05 48 16 00 00 	lea    0x1648(%rip),%rax        # 4040b6 <_IO_stdin_used+0xb6>
  402a6e:	48 89 05 bb 46 00 00 	mov    %rax,0x46bb(%rip)        # 407130 <proc_info_features+0x30>
  402a75:	c7 05 b9 46 00 00 01 	movl   $0x1,0x46b9(%rip)        # 407138 <proc_info_features+0x38>
  402a7c:	00 00 00 
  402a7f:	48 8d 05 34 16 00 00 	lea    0x1634(%rip),%rax        # 4040ba <_IO_stdin_used+0xba>
  402a86:	48 89 05 bb 46 00 00 	mov    %rax,0x46bb(%rip)        # 407148 <proc_info_features+0x48>
  402a8d:	c7 05 b9 46 00 00 02 	movl   $0x2,0x46b9(%rip)        # 407150 <proc_info_features+0x50>
  402a94:	00 00 00 
  402a97:	c7 05 c7 46 00 00 03 	movl   $0x3,0x46c7(%rip)        # 407168 <proc_info_features+0x68>
  402a9e:	00 00 00 
  402aa1:	48 8d 05 17 16 00 00 	lea    0x1617(%rip),%rax        # 4040bf <_IO_stdin_used+0xbf>
  402aa8:	48 89 05 c1 46 00 00 	mov    %rax,0x46c1(%rip)        # 407170 <proc_info_features+0x70>
  402aaf:	48 8d 05 0d 16 00 00 	lea    0x160d(%rip),%rax        # 4040c3 <_IO_stdin_used+0xc3>
  402ab6:	48 89 05 a3 46 00 00 	mov    %rax,0x46a3(%rip)        # 407160 <proc_info_features+0x60>
  402abd:	48 8d 05 03 16 00 00 	lea    0x1603(%rip),%rax        # 4040c7 <_IO_stdin_used+0xc7>
  402ac4:	48 89 05 ad 46 00 00 	mov    %rax,0x46ad(%rip)        # 407178 <proc_info_features+0x78>
  402acb:	c7 05 ab 46 00 00 04 	movl   $0x4,0x46ab(%rip)        # 407180 <proc_info_features+0x80>
  402ad2:	00 00 00 
  402ad5:	c7 05 b9 46 00 00 05 	movl   $0x5,0x46b9(%rip)        # 407198 <proc_info_features+0x98>
  402adc:	00 00 00 
  402adf:	48 8d 05 e8 15 00 00 	lea    0x15e8(%rip),%rax        # 4040ce <_IO_stdin_used+0xce>
  402ae6:	48 89 05 b3 46 00 00 	mov    %rax,0x46b3(%rip)        # 4071a0 <proc_info_features+0xa0>
  402aed:	48 8d 05 de 15 00 00 	lea    0x15de(%rip),%rax        # 4040d2 <_IO_stdin_used+0xd2>
  402af4:	48 89 05 95 46 00 00 	mov    %rax,0x4695(%rip)        # 407190 <proc_info_features+0x90>
  402afb:	c7 05 ab 46 00 00 06 	movl   $0x6,0x46ab(%rip)        # 4071b0 <proc_info_features+0xb0>
  402b02:	00 00 00 
  402b05:	48 8d 05 ca 15 00 00 	lea    0x15ca(%rip),%rax        # 4040d6 <_IO_stdin_used+0xd6>
  402b0c:	48 89 05 a5 46 00 00 	mov    %rax,0x46a5(%rip)        # 4071b8 <proc_info_features+0xb8>
  402b13:	48 8d 05 c1 15 00 00 	lea    0x15c1(%rip),%rax        # 4040db <_IO_stdin_used+0xdb>
  402b1a:	48 89 05 87 46 00 00 	mov    %rax,0x4687(%rip)        # 4071a8 <proc_info_features+0xa8>
  402b21:	c7 05 9d 46 00 00 07 	movl   $0x7,0x469d(%rip)        # 4071c8 <proc_info_features+0xc8>
  402b28:	00 00 00 
  402b2b:	48 8d 05 af 15 00 00 	lea    0x15af(%rip),%rax        # 4040e1 <_IO_stdin_used+0xe1>
  402b32:	48 89 05 97 46 00 00 	mov    %rax,0x4697(%rip)        # 4071d0 <proc_info_features+0xd0>
  402b39:	48 8d 05 a7 15 00 00 	lea    0x15a7(%rip),%rax        # 4040e7 <_IO_stdin_used+0xe7>
  402b40:	48 89 05 79 46 00 00 	mov    %rax,0x4679(%rip)        # 4071c0 <proc_info_features+0xc0>
  402b47:	c7 05 8f 46 00 00 08 	movl   $0x8,0x468f(%rip)        # 4071e0 <proc_info_features+0xe0>
  402b4e:	00 00 00 
  402b51:	48 8d 05 88 15 00 00 	lea    0x1588(%rip),%rax        # 4040e0 <_IO_stdin_used+0xe0>
  402b58:	48 89 05 89 46 00 00 	mov    %rax,0x4689(%rip)        # 4071e8 <proc_info_features+0xe8>
  402b5f:	48 8d 05 80 15 00 00 	lea    0x1580(%rip),%rax        # 4040e6 <_IO_stdin_used+0xe6>
  402b66:	48 89 05 6b 46 00 00 	mov    %rax,0x466b(%rip)        # 4071d8 <proc_info_features+0xd8>
  402b6d:	c7 05 81 46 00 00 09 	movl   $0x9,0x4681(%rip)        # 4071f8 <proc_info_features+0xf8>
  402b74:	00 00 00 
  402b77:	48 8d 05 6e 15 00 00 	lea    0x156e(%rip),%rax        # 4040ec <_IO_stdin_used+0xec>
  402b7e:	48 89 05 7b 46 00 00 	mov    %rax,0x467b(%rip)        # 407200 <proc_info_features+0x100>
  402b85:	48 8d 05 67 15 00 00 	lea    0x1567(%rip),%rax        # 4040f3 <_IO_stdin_used+0xf3>
  402b8c:	48 89 05 5d 46 00 00 	mov    %rax,0x465d(%rip)        # 4071f0 <proc_info_features+0xf0>
  402b93:	c7 05 73 46 00 00 0a 	movl   $0xa,0x4673(%rip)        # 407210 <proc_info_features+0x110>
  402b9a:	00 00 00 
  402b9d:	48 8d 05 56 15 00 00 	lea    0x1556(%rip),%rax        # 4040fa <_IO_stdin_used+0xfa>
  402ba4:	48 89 05 6d 46 00 00 	mov    %rax,0x466d(%rip)        # 407218 <proc_info_features+0x118>
  402bab:	48 8d 05 4d 15 00 00 	lea    0x154d(%rip),%rax        # 4040ff <_IO_stdin_used+0xff>
  402bb2:	48 89 05 4f 46 00 00 	mov    %rax,0x464f(%rip)        # 407208 <proc_info_features+0x108>
  402bb9:	c7 05 65 46 00 00 0b 	movl   $0xb,0x4665(%rip)        # 407228 <proc_info_features+0x128>
  402bc0:	00 00 00 
  402bc3:	48 8d 05 3c 15 00 00 	lea    0x153c(%rip),%rax        # 404106 <_IO_stdin_used+0x106>
  402bca:	48 89 05 5f 46 00 00 	mov    %rax,0x465f(%rip)        # 407230 <proc_info_features+0x130>
  402bd1:	48 8d 05 34 15 00 00 	lea    0x1534(%rip),%rax        # 40410c <_IO_stdin_used+0x10c>
  402bd8:	48 89 05 41 46 00 00 	mov    %rax,0x4641(%rip)        # 407220 <proc_info_features+0x120>
  402bdf:	c7 05 57 46 00 00 0c 	movl   $0xc,0x4657(%rip)        # 407240 <proc_info_features+0x140>
  402be6:	00 00 00 
  402be9:	48 8d 05 22 15 00 00 	lea    0x1522(%rip),%rax        # 404112 <_IO_stdin_used+0x112>
  402bf0:	48 89 05 51 46 00 00 	mov    %rax,0x4651(%rip)        # 407248 <proc_info_features+0x148>
  402bf7:	48 8d 05 1b 15 00 00 	lea    0x151b(%rip),%rax        # 404119 <_IO_stdin_used+0x119>
  402bfe:	48 89 05 33 46 00 00 	mov    %rax,0x4633(%rip)        # 407238 <proc_info_features+0x138>
  402c05:	c7 05 49 46 00 00 0d 	movl   $0xd,0x4649(%rip)        # 407258 <proc_info_features+0x158>
  402c0c:	00 00 00 
  402c0f:	48 8d 05 0a 15 00 00 	lea    0x150a(%rip),%rax        # 404120 <_IO_stdin_used+0x120>
  402c16:	48 89 05 43 46 00 00 	mov    %rax,0x4643(%rip)        # 407260 <proc_info_features+0x160>
  402c1d:	48 8d 05 dc 16 00 00 	lea    0x16dc(%rip),%rax        # 404300 <_IO_stdin_used+0x300>
  402c24:	48 89 05 25 46 00 00 	mov    %rax,0x4625(%rip)        # 407250 <proc_info_features+0x150>
  402c2b:	c7 05 3b 46 00 00 0e 	movl   $0xe,0x463b(%rip)        # 407270 <proc_info_features+0x170>
  402c32:	00 00 00 
  402c35:	48 8d 05 af 16 00 00 	lea    0x16af(%rip),%rax        # 4042eb <_IO_stdin_used+0x2eb>
  402c3c:	48 89 05 35 46 00 00 	mov    %rax,0x4635(%rip)        # 407278 <proc_info_features+0x178>
  402c43:	48 8d 05 a6 16 00 00 	lea    0x16a6(%rip),%rax        # 4042f0 <_IO_stdin_used+0x2f0>
  402c4a:	48 89 05 17 46 00 00 	mov    %rax,0x4617(%rip)        # 407268 <proc_info_features+0x168>
  402c51:	c7 05 2d 46 00 00 10 	movl   $0x10,0x462d(%rip)        # 407288 <proc_info_features+0x188>
  402c58:	00 00 00 
  402c5b:	48 8d 05 c5 14 00 00 	lea    0x14c5(%rip),%rax        # 404127 <_IO_stdin_used+0x127>
  402c62:	48 89 05 27 46 00 00 	mov    %rax,0x4627(%rip)        # 407290 <proc_info_features+0x190>
  402c69:	48 8d 05 bb 14 00 00 	lea    0x14bb(%rip),%rax        # 40412b <_IO_stdin_used+0x12b>
  402c70:	48 89 05 09 46 00 00 	mov    %rax,0x4609(%rip)        # 407280 <proc_info_features+0x180>
  402c77:	c7 05 1f 46 00 00 0f 	movl   $0xf,0x461f(%rip)        # 4072a0 <proc_info_features+0x1a0>
  402c7e:	00 00 00 
  402c81:	48 8d 05 a7 14 00 00 	lea    0x14a7(%rip),%rax        # 40412f <_IO_stdin_used+0x12f>
  402c88:	48 89 05 19 46 00 00 	mov    %rax,0x4619(%rip)        # 4072a8 <proc_info_features+0x1a8>
  402c8f:	48 8d 05 9e 14 00 00 	lea    0x149e(%rip),%rax        # 404134 <_IO_stdin_used+0x134>
  402c96:	48 89 05 fb 45 00 00 	mov    %rax,0x45fb(%rip)        # 407298 <proc_info_features+0x198>
  402c9d:	c7 05 11 46 00 00 11 	movl   $0x11,0x4611(%rip)        # 4072b8 <proc_info_features+0x1b8>
  402ca4:	00 00 00 
  402ca7:	48 8d 05 8b 14 00 00 	lea    0x148b(%rip),%rax        # 404139 <_IO_stdin_used+0x139>
  402cae:	48 89 05 0b 46 00 00 	mov    %rax,0x460b(%rip)        # 4072c0 <proc_info_features+0x1c0>
  402cb5:	48 8d 05 83 14 00 00 	lea    0x1483(%rip),%rax        # 40413f <_IO_stdin_used+0x13f>
  402cbc:	48 89 05 ed 45 00 00 	mov    %rax,0x45ed(%rip)        # 4072b0 <proc_info_features+0x1b0>
  402cc3:	c7 05 03 46 00 00 12 	movl   $0x12,0x4603(%rip)        # 4072d0 <proc_info_features+0x1d0>
  402cca:	00 00 00 
  402ccd:	48 8d 05 f0 14 00 00 	lea    0x14f0(%rip),%rax        # 4041c4 <_IO_stdin_used+0x1c4>
  402cd4:	48 89 05 fd 45 00 00 	mov    %rax,0x45fd(%rip)        # 4072d8 <proc_info_features+0x1d8>
  402cdb:	48 8d 05 63 14 00 00 	lea    0x1463(%rip),%rax        # 404145 <_IO_stdin_used+0x145>
  402ce2:	48 89 05 df 45 00 00 	mov    %rax,0x45df(%rip)        # 4072c8 <proc_info_features+0x1c8>
  402ce9:	c7 05 f5 45 00 00 13 	movl   $0x13,0x45f5(%rip)        # 4072e8 <proc_info_features+0x1e8>
  402cf0:	00 00 00 
  402cf3:	48 8d 05 4c 15 00 00 	lea    0x154c(%rip),%rax        # 404246 <_IO_stdin_used+0x246>
  402cfa:	48 89 05 ef 45 00 00 	mov    %rax,0x45ef(%rip)        # 4072f0 <proc_info_features+0x1f0>
  402d01:	48 8d 05 49 15 00 00 	lea    0x1549(%rip),%rax        # 404251 <_IO_stdin_used+0x251>
  402d08:	48 89 05 d1 45 00 00 	mov    %rax,0x45d1(%rip)        # 4072e0 <proc_info_features+0x1e0>
  402d0f:	c7 05 e7 45 00 00 14 	movl   $0x14,0x45e7(%rip)        # 407300 <proc_info_features+0x200>
  402d16:	00 00 00 
  402d19:	48 8d 05 29 14 00 00 	lea    0x1429(%rip),%rax        # 404149 <_IO_stdin_used+0x149>
  402d20:	48 89 05 e1 45 00 00 	mov    %rax,0x45e1(%rip)        # 407308 <proc_info_features+0x208>
  402d27:	48 8d 05 21 14 00 00 	lea    0x1421(%rip),%rax        # 40414f <_IO_stdin_used+0x14f>
  402d2e:	48 89 05 c3 45 00 00 	mov    %rax,0x45c3(%rip)        # 4072f8 <proc_info_features+0x1f8>
  402d35:	c7 05 d9 45 00 00 15 	movl   $0x15,0x45d9(%rip)        # 407318 <proc_info_features+0x218>
  402d3c:	00 00 00 
  402d3f:	48 8d 05 0f 14 00 00 	lea    0x140f(%rip),%rax        # 404155 <_IO_stdin_used+0x155>
  402d46:	48 89 05 d3 45 00 00 	mov    %rax,0x45d3(%rip)        # 407320 <proc_info_features+0x220>
  402d4d:	48 8d 05 05 14 00 00 	lea    0x1405(%rip),%rax        # 404159 <_IO_stdin_used+0x159>
  402d54:	48 89 05 b5 45 00 00 	mov    %rax,0x45b5(%rip)        # 407310 <proc_info_features+0x210>
  402d5b:	c7 05 cb 45 00 00 16 	movl   $0x16,0x45cb(%rip)        # 407330 <proc_info_features+0x230>
  402d62:	00 00 00 
  402d65:	48 8d 05 f1 13 00 00 	lea    0x13f1(%rip),%rax        # 40415d <_IO_stdin_used+0x15d>
  402d6c:	48 89 05 c5 45 00 00 	mov    %rax,0x45c5(%rip)        # 407338 <proc_info_features+0x238>
  402d73:	48 8d 05 e7 13 00 00 	lea    0x13e7(%rip),%rax        # 404161 <_IO_stdin_used+0x161>
  402d7a:	48 89 05 a7 45 00 00 	mov    %rax,0x45a7(%rip)        # 407328 <proc_info_features+0x228>
  402d81:	c7 05 bd 45 00 00 17 	movl   $0x17,0x45bd(%rip)        # 407348 <proc_info_features+0x248>
  402d88:	00 00 00 
  402d8b:	48 8d 05 d3 13 00 00 	lea    0x13d3(%rip),%rax        # 404165 <_IO_stdin_used+0x165>
  402d92:	48 89 05 b7 45 00 00 	mov    %rax,0x45b7(%rip)        # 407350 <proc_info_features+0x250>
  402d99:	48 8d 05 ca 13 00 00 	lea    0x13ca(%rip),%rax        # 40416a <_IO_stdin_used+0x16a>
  402da0:	48 89 05 99 45 00 00 	mov    %rax,0x4599(%rip)        # 407340 <proc_info_features+0x240>
  402da7:	c7 05 af 45 00 00 1b 	movl   $0x1b,0x45af(%rip)        # 407360 <proc_info_features+0x260>
  402dae:	00 00 00 
  402db1:	48 8d 05 b7 13 00 00 	lea    0x13b7(%rip),%rax        # 40416f <_IO_stdin_used+0x16f>
  402db8:	48 89 05 a9 45 00 00 	mov    %rax,0x45a9(%rip)        # 407368 <proc_info_features+0x268>
  402dbf:	48 8d 05 b1 13 00 00 	lea    0x13b1(%rip),%rax        # 404177 <_IO_stdin_used+0x177>
  402dc6:	48 89 05 8b 45 00 00 	mov    %rax,0x458b(%rip)        # 407358 <proc_info_features+0x258>
  402dcd:	c7 05 a1 45 00 00 18 	movl   $0x18,0x45a1(%rip)        # 407378 <proc_info_features+0x278>
  402dd4:	00 00 00 
  402dd7:	48 8d 05 a1 13 00 00 	lea    0x13a1(%rip),%rax        # 40417f <_IO_stdin_used+0x17f>
  402dde:	48 89 05 9b 45 00 00 	mov    %rax,0x459b(%rip)        # 407380 <proc_info_features+0x280>
  402de5:	48 8d 05 9c 13 00 00 	lea    0x139c(%rip),%rax        # 404188 <_IO_stdin_used+0x188>
  402dec:	48 89 05 7d 45 00 00 	mov    %rax,0x457d(%rip)        # 407370 <proc_info_features+0x270>
  402df3:	c7 05 93 45 00 00 19 	movl   $0x19,0x4593(%rip)        # 407390 <proc_info_features+0x290>
  402dfa:	00 00 00 
  402dfd:	48 8d 05 8d 13 00 00 	lea    0x138d(%rip),%rax        # 404191 <_IO_stdin_used+0x191>
  402e04:	48 89 05 8d 45 00 00 	mov    %rax,0x458d(%rip)        # 407398 <proc_info_features+0x298>
  402e0b:	48 8d 05 87 13 00 00 	lea    0x1387(%rip),%rax        # 404199 <_IO_stdin_used+0x199>
  402e12:	48 89 05 6f 45 00 00 	mov    %rax,0x456f(%rip)        # 407388 <proc_info_features+0x288>
  402e19:	48 8d 05 81 13 00 00 	lea    0x1381(%rip),%rax        # 4041a1 <_IO_stdin_used+0x1a1>
  402e20:	48 89 05 79 45 00 00 	mov    %rax,0x4579(%rip)        # 4073a0 <proc_info_features+0x2a0>
  402e27:	c7 05 77 45 00 00 1a 	movl   $0x1a,0x4577(%rip)        # 4073a8 <proc_info_features+0x2a8>
  402e2e:	00 00 00 
  402e31:	c7 05 85 45 00 00 1c 	movl   $0x1c,0x4585(%rip)        # 4073c0 <proc_info_features+0x2c0>
  402e38:	00 00 00 
  402e3b:	48 8d 05 65 13 00 00 	lea    0x1365(%rip),%rax        # 4041a7 <_IO_stdin_used+0x1a7>
  402e42:	48 89 05 7f 45 00 00 	mov    %rax,0x457f(%rip)        # 4073c8 <proc_info_features+0x2c8>
  402e49:	48 8d 05 5b 13 00 00 	lea    0x135b(%rip),%rax        # 4041ab <_IO_stdin_used+0x1ab>
  402e50:	48 89 05 61 45 00 00 	mov    %rax,0x4561(%rip)        # 4073b8 <proc_info_features+0x2b8>
  402e57:	c7 05 77 45 00 00 1d 	movl   $0x1d,0x4577(%rip)        # 4073d8 <proc_info_features+0x2d8>
  402e5e:	00 00 00 
  402e61:	48 8d 05 47 13 00 00 	lea    0x1347(%rip),%rax        # 4041af <_IO_stdin_used+0x1af>
  402e68:	48 89 05 71 45 00 00 	mov    %rax,0x4571(%rip)        # 4073e0 <proc_info_features+0x2e0>
  402e6f:	48 8d 05 40 13 00 00 	lea    0x1340(%rip),%rax        # 4041b6 <_IO_stdin_used+0x1b6>
  402e76:	48 89 05 53 45 00 00 	mov    %rax,0x4553(%rip)        # 4073d0 <proc_info_features+0x2d0>
  402e7d:	c7 05 69 45 00 00 1e 	movl   $0x1e,0x4569(%rip)        # 4073f0 <proc_info_features+0x2f0>
  402e84:	00 00 00 
  402e87:	48 8d 05 2f 13 00 00 	lea    0x132f(%rip),%rax        # 4041bd <_IO_stdin_used+0x1bd>
  402e8e:	48 89 05 63 45 00 00 	mov    %rax,0x4563(%rip)        # 4073f8 <proc_info_features+0x2f8>
  402e95:	48 8d 05 2c 13 00 00 	lea    0x132c(%rip),%rax        # 4041c8 <_IO_stdin_used+0x1c8>
  402e9c:	48 89 05 45 45 00 00 	mov    %rax,0x4545(%rip)        # 4073e8 <proc_info_features+0x2e8>
  402ea3:	c7 05 5b 45 00 00 ff 	movl   $0xffffffff,0x455b(%rip)        # 407408 <proc_info_features+0x308>
  402eaa:	ff ff ff 
  402ead:	c7 05 69 45 00 00 20 	movl   $0x20,0x4569(%rip)        # 407420 <proc_info_features+0x320>
  402eb4:	00 00 00 
  402eb7:	48 8d 05 17 13 00 00 	lea    0x1317(%rip),%rax        # 4041d5 <_IO_stdin_used+0x1d5>
  402ebe:	48 89 05 63 45 00 00 	mov    %rax,0x4563(%rip)        # 407428 <proc_info_features+0x328>
  402ec5:	48 8d 05 12 13 00 00 	lea    0x1312(%rip),%rax        # 4041de <_IO_stdin_used+0x1de>
  402ecc:	48 89 05 45 45 00 00 	mov    %rax,0x4545(%rip)        # 407418 <proc_info_features+0x318>
  402ed3:	c7 05 5b 45 00 00 21 	movl   $0x21,0x455b(%rip)        # 407438 <proc_info_features+0x338>
  402eda:	00 00 00 
  402edd:	48 8d 05 03 13 00 00 	lea    0x1303(%rip),%rax        # 4041e7 <_IO_stdin_used+0x1e7>
  402ee4:	48 89 05 55 45 00 00 	mov    %rax,0x4555(%rip)        # 407440 <proc_info_features+0x340>
  402eeb:	48 8d 05 fe 12 00 00 	lea    0x12fe(%rip),%rax        # 4041f0 <_IO_stdin_used+0x1f0>
  402ef2:	48 89 05 37 45 00 00 	mov    %rax,0x4537(%rip)        # 407430 <proc_info_features+0x330>
  402ef9:	c7 05 4d 45 00 00 22 	movl   $0x22,0x454d(%rip)        # 407450 <proc_info_features+0x350>
  402f00:	00 00 00 
  402f03:	48 8d 05 ef 12 00 00 	lea    0x12ef(%rip),%rax        # 4041f9 <_IO_stdin_used+0x1f9>
  402f0a:	48 89 05 47 45 00 00 	mov    %rax,0x4547(%rip)        # 407458 <proc_info_features+0x358>
  402f11:	48 8d 05 ea 12 00 00 	lea    0x12ea(%rip),%rax        # 404202 <_IO_stdin_used+0x202>
  402f18:	48 89 05 29 45 00 00 	mov    %rax,0x4529(%rip)        # 407448 <proc_info_features+0x348>
  402f1f:	c7 05 3f 45 00 00 23 	movl   $0x23,0x453f(%rip)        # 407468 <proc_info_features+0x368>
  402f26:	00 00 00 
  402f29:	48 8d 05 db 12 00 00 	lea    0x12db(%rip),%rax        # 40420b <_IO_stdin_used+0x20b>
  402f30:	48 89 05 39 45 00 00 	mov    %rax,0x4539(%rip)        # 407470 <proc_info_features+0x370>
  402f37:	48 8d 05 d1 12 00 00 	lea    0x12d1(%rip),%rax        # 40420f <_IO_stdin_used+0x20f>
  402f3e:	48 89 05 1b 45 00 00 	mov    %rax,0x451b(%rip)        # 407460 <proc_info_features+0x360>
  402f45:	c7 05 31 45 00 00 24 	movl   $0x24,0x4531(%rip)        # 407480 <proc_info_features+0x380>
  402f4c:	00 00 00 
  402f4f:	48 8d 05 bd 12 00 00 	lea    0x12bd(%rip),%rax        # 404213 <_IO_stdin_used+0x213>
  402f56:	48 89 05 2b 45 00 00 	mov    %rax,0x452b(%rip)        # 407488 <proc_info_features+0x388>
  402f5d:	48 8d 05 b3 12 00 00 	lea    0x12b3(%rip),%rax        # 404217 <_IO_stdin_used+0x217>
  402f64:	48 89 05 0d 45 00 00 	mov    %rax,0x450d(%rip)        # 407478 <proc_info_features+0x378>
  402f6b:	c7 05 23 45 00 00 25 	movl   $0x25,0x4523(%rip)        # 407498 <proc_info_features+0x398>
  402f72:	00 00 00 
  402f75:	48 8d 05 9f 12 00 00 	lea    0x129f(%rip),%rax        # 40421b <_IO_stdin_used+0x21b>
  402f7c:	48 89 05 1d 45 00 00 	mov    %rax,0x451d(%rip)        # 4074a0 <proc_info_features+0x3a0>
  402f83:	48 8d 05 9a 12 00 00 	lea    0x129a(%rip),%rax        # 404224 <_IO_stdin_used+0x224>
  402f8a:	48 89 05 ff 44 00 00 	mov    %rax,0x44ff(%rip)        # 407490 <proc_info_features+0x390>
  402f91:	c7 05 15 45 00 00 26 	movl   $0x26,0x4515(%rip)        # 4074b0 <proc_info_features+0x3b0>
  402f98:	00 00 00 
  402f9b:	48 8d 05 8b 12 00 00 	lea    0x128b(%rip),%rax        # 40422d <_IO_stdin_used+0x22d>
  402fa2:	48 89 05 0f 45 00 00 	mov    %rax,0x450f(%rip)        # 4074b8 <proc_info_features+0x3b8>
  402fa9:	48 8d 05 86 12 00 00 	lea    0x1286(%rip),%rax        # 404236 <_IO_stdin_used+0x236>
  402fb0:	48 89 05 f1 44 00 00 	mov    %rax,0x44f1(%rip)        # 4074a8 <proc_info_features+0x3a8>
  402fb7:	c7 05 07 45 00 00 27 	movl   $0x27,0x4507(%rip)        # 4074c8 <proc_info_features+0x3c8>
  402fbe:	00 00 00 
  402fc1:	48 8d 05 77 12 00 00 	lea    0x1277(%rip),%rax        # 40423f <_IO_stdin_used+0x23f>
  402fc8:	48 89 05 01 45 00 00 	mov    %rax,0x4501(%rip)        # 4074d0 <proc_info_features+0x3d0>
  402fcf:	48 8d 05 74 12 00 00 	lea    0x1274(%rip),%rax        # 40424a <_IO_stdin_used+0x24a>
  402fd6:	48 89 05 e3 44 00 00 	mov    %rax,0x44e3(%rip)        # 4074c0 <proc_info_features+0x3c0>
  402fdd:	c7 05 f9 44 00 00 28 	movl   $0x28,0x44f9(%rip)        # 4074e0 <proc_info_features+0x3e0>
  402fe4:	00 00 00 
  402fe7:	48 8d 05 67 12 00 00 	lea    0x1267(%rip),%rax        # 404255 <_IO_stdin_used+0x255>
  402fee:	48 89 05 f3 44 00 00 	mov    %rax,0x44f3(%rip)        # 4074e8 <proc_info_features+0x3e8>
  402ff5:	48 8d 05 66 12 00 00 	lea    0x1266(%rip),%rax        # 404262 <_IO_stdin_used+0x262>
  402ffc:	48 89 05 d5 44 00 00 	mov    %rax,0x44d5(%rip)        # 4074d8 <proc_info_features+0x3d8>
  403003:	c7 05 eb 44 00 00 29 	movl   $0x29,0x44eb(%rip)        # 4074f8 <proc_info_features+0x3f8>
  40300a:	00 00 00 
  40300d:	48 8d 05 5c 12 00 00 	lea    0x125c(%rip),%rax        # 404270 <_IO_stdin_used+0x270>
  403014:	48 89 05 e5 44 00 00 	mov    %rax,0x44e5(%rip)        # 407500 <proc_info_features+0x400>
  40301b:	48 8d 05 5b 12 00 00 	lea    0x125b(%rip),%rax        # 40427d <_IO_stdin_used+0x27d>
  403022:	48 89 05 c7 44 00 00 	mov    %rax,0x44c7(%rip)        # 4074f0 <proc_info_features+0x3f0>
  403029:	c7 05 dd 44 00 00 2a 	movl   $0x2a,0x44dd(%rip)        # 407510 <proc_info_features+0x410>
  403030:	00 00 00 
  403033:	48 8d 05 51 12 00 00 	lea    0x1251(%rip),%rax        # 40428b <_IO_stdin_used+0x28b>
  40303a:	48 89 05 d7 44 00 00 	mov    %rax,0x44d7(%rip)        # 407518 <proc_info_features+0x418>
  403041:	48 8d 05 53 12 00 00 	lea    0x1253(%rip),%rax        # 40429b <_IO_stdin_used+0x29b>
  403048:	48 89 05 b9 44 00 00 	mov    %rax,0x44b9(%rip)        # 407508 <proc_info_features+0x408>
  40304f:	c7 05 cf 44 00 00 2b 	movl   $0x2b,0x44cf(%rip)        # 407528 <proc_info_features+0x428>
  403056:	00 00 00 
  403059:	48 8d 05 4c 12 00 00 	lea    0x124c(%rip),%rax        # 4042ac <_IO_stdin_used+0x2ac>
  403060:	48 89 05 c9 44 00 00 	mov    %rax,0x44c9(%rip)        # 407530 <proc_info_features+0x430>
  403067:	48 8d 05 4b 12 00 00 	lea    0x124b(%rip),%rax        # 4042b9 <_IO_stdin_used+0x2b9>
  40306e:	48 89 05 ab 44 00 00 	mov    %rax,0x44ab(%rip)        # 407520 <proc_info_features+0x420>
  403075:	c7 05 c1 44 00 00 2c 	movl   $0x2c,0x44c1(%rip)        # 407540 <proc_info_features+0x440>
  40307c:	00 00 00 
  40307f:	48 8d 05 41 12 00 00 	lea    0x1241(%rip),%rax        # 4042c7 <_IO_stdin_used+0x2c7>
  403086:	48 89 05 bb 44 00 00 	mov    %rax,0x44bb(%rip)        # 407548 <proc_info_features+0x448>
  40308d:	48 8d 05 3f 12 00 00 	lea    0x123f(%rip),%rax        # 4042d3 <_IO_stdin_used+0x2d3>
  403094:	48 89 05 9d 44 00 00 	mov    %rax,0x449d(%rip)        # 407538 <proc_info_features+0x438>
  40309b:	c7 05 b3 44 00 00 2d 	movl   $0x2d,0x44b3(%rip)        # 407558 <proc_info_features+0x458>
  4030a2:	00 00 00 
  4030a5:	48 8d 05 34 12 00 00 	lea    0x1234(%rip),%rax        # 4042e0 <_IO_stdin_used+0x2e0>
  4030ac:	48 89 05 ad 44 00 00 	mov    %rax,0x44ad(%rip)        # 407560 <proc_info_features+0x460>
  4030b3:	48 8d 05 2b 12 00 00 	lea    0x122b(%rip),%rax        # 4042e5 <_IO_stdin_used+0x2e5>
  4030ba:	48 89 05 8f 44 00 00 	mov    %rax,0x448f(%rip)        # 407550 <proc_info_features+0x450>
  4030c1:	c7 05 a5 44 00 00 2e 	movl   $0x2e,0x44a5(%rip)        # 407570 <proc_info_features+0x470>
  4030c8:	00 00 00 
  4030cb:	48 8d 05 18 12 00 00 	lea    0x1218(%rip),%rax        # 4042ea <_IO_stdin_used+0x2ea>
  4030d2:	48 89 05 9f 44 00 00 	mov    %rax,0x449f(%rip)        # 407578 <proc_info_features+0x478>
  4030d9:	48 8d 05 0f 12 00 00 	lea    0x120f(%rip),%rax        # 4042ef <_IO_stdin_used+0x2ef>
  4030e0:	48 89 05 81 44 00 00 	mov    %rax,0x4481(%rip)        # 407568 <proc_info_features+0x468>
  4030e7:	c7 05 97 44 00 00 2f 	movl   $0x2f,0x4497(%rip)        # 407588 <proc_info_features+0x488>
  4030ee:	00 00 00 
  4030f1:	48 8d 05 fc 11 00 00 	lea    0x11fc(%rip),%rax        # 4042f4 <_IO_stdin_used+0x2f4>
  4030f8:	48 89 05 91 44 00 00 	mov    %rax,0x4491(%rip)        # 407590 <proc_info_features+0x490>
  4030ff:	48 8d 05 f9 11 00 00 	lea    0x11f9(%rip),%rax        # 4042ff <_IO_stdin_used+0x2ff>
  403106:	48 89 05 73 44 00 00 	mov    %rax,0x4473(%rip)        # 407580 <proc_info_features+0x480>
  40310d:	c7 05 89 44 00 00 30 	movl   $0x30,0x4489(%rip)        # 4075a0 <proc_info_features+0x4a0>
  403114:	00 00 00 
  403117:	48 8d 05 ec 11 00 00 	lea    0x11ec(%rip),%rax        # 40430a <_IO_stdin_used+0x30a>
  40311e:	48 89 05 83 44 00 00 	mov    %rax,0x4483(%rip)        # 4075a8 <proc_info_features+0x4a8>
  403125:	48 8d 05 e9 11 00 00 	lea    0x11e9(%rip),%rax        # 404315 <_IO_stdin_used+0x315>
  40312c:	48 89 05 65 44 00 00 	mov    %rax,0x4465(%rip)        # 407598 <proc_info_features+0x498>
  403133:	c7 05 7b 44 00 00 31 	movl   $0x31,0x447b(%rip)        # 4075b8 <proc_info_features+0x4b8>
  40313a:	00 00 00 
  40313d:	48 8d 05 dd 11 00 00 	lea    0x11dd(%rip),%rax        # 404321 <_IO_stdin_used+0x321>
  403144:	48 89 05 75 44 00 00 	mov    %rax,0x4475(%rip)        # 4075c0 <proc_info_features+0x4c0>
  40314b:	48 8d 05 d4 11 00 00 	lea    0x11d4(%rip),%rax        # 404326 <_IO_stdin_used+0x326>
  403152:	48 89 05 57 44 00 00 	mov    %rax,0x4457(%rip)        # 4075b0 <proc_info_features+0x4b0>
  403159:	c7 05 6d 44 00 00 32 	movl   $0x32,0x446d(%rip)        # 4075d0 <proc_info_features+0x4d0>
  403160:	00 00 00 
  403163:	48 8d 05 c1 11 00 00 	lea    0x11c1(%rip),%rax        # 40432b <_IO_stdin_used+0x32b>
  40316a:	48 89 05 67 44 00 00 	mov    %rax,0x4467(%rip)        # 4075d8 <proc_info_features+0x4d8>
  403171:	48 8d 05 b9 11 00 00 	lea    0x11b9(%rip),%rax        # 404331 <_IO_stdin_used+0x331>
  403178:	48 89 05 49 44 00 00 	mov    %rax,0x4449(%rip)        # 4075c8 <proc_info_features+0x4c8>
  40317f:	c7 05 5f 44 00 00 33 	movl   $0x33,0x445f(%rip)        # 4075e8 <proc_info_features+0x4e8>
  403186:	00 00 00 
  403189:	48 8d 05 a7 11 00 00 	lea    0x11a7(%rip),%rax        # 404337 <_IO_stdin_used+0x337>
  403190:	48 89 05 59 44 00 00 	mov    %rax,0x4459(%rip)        # 4075f0 <proc_info_features+0x4f0>
  403197:	48 8d 05 9d 11 00 00 	lea    0x119d(%rip),%rax        # 40433b <_IO_stdin_used+0x33b>
  40319e:	48 89 05 3b 44 00 00 	mov    %rax,0x443b(%rip)        # 4075e0 <proc_info_features+0x4e0>
  4031a5:	c7 05 51 44 00 00 34 	movl   $0x34,0x4451(%rip)        # 407600 <proc_info_features+0x500>
  4031ac:	00 00 00 
  4031af:	48 8d 05 89 11 00 00 	lea    0x1189(%rip),%rax        # 40433f <_IO_stdin_used+0x33f>
  4031b6:	48 89 05 4b 44 00 00 	mov    %rax,0x444b(%rip)        # 407608 <proc_info_features+0x508>
  4031bd:	48 8d 05 81 11 00 00 	lea    0x1181(%rip),%rax        # 404345 <_IO_stdin_used+0x345>
  4031c4:	48 89 05 2d 44 00 00 	mov    %rax,0x442d(%rip)        # 4075f8 <proc_info_features+0x4f8>
  4031cb:	c7 05 43 44 00 00 35 	movl   $0x35,0x4443(%rip)        # 407618 <proc_info_features+0x518>
  4031d2:	00 00 00 
  4031d5:	48 8d 05 6f 11 00 00 	lea    0x116f(%rip),%rax        # 40434b <_IO_stdin_used+0x34b>
  4031dc:	48 89 05 3d 44 00 00 	mov    %rax,0x443d(%rip)        # 407620 <proc_info_features+0x520>
  4031e3:	48 8d 05 65 11 00 00 	lea    0x1165(%rip),%rax        # 40434f <_IO_stdin_used+0x34f>
  4031ea:	48 89 05 1f 44 00 00 	mov    %rax,0x441f(%rip)        # 407610 <proc_info_features+0x510>
  4031f1:	c7 05 35 44 00 00 36 	movl   $0x36,0x4435(%rip)        # 407630 <proc_info_features+0x530>
  4031f8:	00 00 00 
  4031fb:	48 8d 05 51 11 00 00 	lea    0x1151(%rip),%rax        # 404353 <_IO_stdin_used+0x353>
  403202:	48 89 05 2f 44 00 00 	mov    %rax,0x442f(%rip)        # 407638 <proc_info_features+0x538>
  403209:	48 8d 05 4c 11 00 00 	lea    0x114c(%rip),%rax        # 40435c <_IO_stdin_used+0x35c>
  403210:	48 89 05 11 44 00 00 	mov    %rax,0x4411(%rip)        # 407628 <proc_info_features+0x528>
  403217:	c7 05 27 44 00 00 37 	movl   $0x37,0x4427(%rip)        # 407648 <proc_info_features+0x548>
  40321e:	00 00 00 
  403221:	48 8d 05 3d 11 00 00 	lea    0x113d(%rip),%rax        # 404365 <_IO_stdin_used+0x365>
  403228:	48 89 05 21 44 00 00 	mov    %rax,0x4421(%rip)        # 407650 <proc_info_features+0x550>
  40322f:	48 8d 05 37 11 00 00 	lea    0x1137(%rip),%rax        # 40436d <_IO_stdin_used+0x36d>
  403236:	48 89 05 03 44 00 00 	mov    %rax,0x4403(%rip)        # 407640 <proc_info_features+0x540>
  40323d:	c7 05 19 44 00 00 38 	movl   $0x38,0x4419(%rip)        # 407660 <proc_info_features+0x560>
  403244:	00 00 00 
  403247:	48 8d 05 27 11 00 00 	lea    0x1127(%rip),%rax        # 404375 <_IO_stdin_used+0x375>
  40324e:	48 89 05 13 44 00 00 	mov    %rax,0x4413(%rip)        # 407668 <proc_info_features+0x568>
  403255:	48 8d 05 2c 11 00 00 	lea    0x112c(%rip),%rax        # 404388 <_IO_stdin_used+0x388>
  40325c:	48 89 05 f5 43 00 00 	mov    %rax,0x43f5(%rip)        # 407658 <proc_info_features+0x558>
  403263:	c7 05 0b 44 00 00 3c 	movl   $0x3c,0x440b(%rip)        # 407678 <proc_info_features+0x578>
  40326a:	00 00 00 
  40326d:	48 8d 05 28 11 00 00 	lea    0x1128(%rip),%rax        # 40439c <_IO_stdin_used+0x39c>
  403274:	48 89 05 05 44 00 00 	mov    %rax,0x4405(%rip)        # 407680 <proc_info_features+0x580>
  40327b:	48 8d 05 25 11 00 00 	lea    0x1125(%rip),%rax        # 4043a7 <_IO_stdin_used+0x3a7>
  403282:	48 89 05 e7 43 00 00 	mov    %rax,0x43e7(%rip)        # 407670 <proc_info_features+0x570>
  403289:	c7 05 fd 43 00 00 40 	movl   $0x40,0x43fd(%rip)        # 407690 <proc_info_features+0x590>
  403290:	00 00 00 
  403293:	48 8d 05 19 11 00 00 	lea    0x1119(%rip),%rax        # 4043b3 <_IO_stdin_used+0x3b3>
  40329a:	48 89 05 f7 43 00 00 	mov    %rax,0x43f7(%rip)        # 407698 <proc_info_features+0x598>
  4032a1:	48 8d 05 14 11 00 00 	lea    0x1114(%rip),%rax        # 4043bc <_IO_stdin_used+0x3bc>
  4032a8:	48 89 05 d9 43 00 00 	mov    %rax,0x43d9(%rip)        # 407688 <proc_info_features+0x588>
  4032af:	c7 05 ef 43 00 00 41 	movl   $0x41,0x43ef(%rip)        # 4076a8 <proc_info_features+0x5a8>
  4032b6:	00 00 00 
  4032b9:	48 8d 05 05 11 00 00 	lea    0x1105(%rip),%rax        # 4043c5 <_IO_stdin_used+0x3c5>
  4032c0:	48 89 05 e9 43 00 00 	mov    %rax,0x43e9(%rip)        # 4076b0 <proc_info_features+0x5b0>
  4032c7:	48 8d 05 ff 10 00 00 	lea    0x10ff(%rip),%rax        # 4043cd <_IO_stdin_used+0x3cd>
  4032ce:	48 89 05 cb 43 00 00 	mov    %rax,0x43cb(%rip)        # 4076a0 <proc_info_features+0x5a0>
  4032d5:	c7 05 e1 43 00 00 42 	movl   $0x42,0x43e1(%rip)        # 4076c0 <proc_info_features+0x5c0>
  4032dc:	00 00 00 
  4032df:	48 8d 05 ef 10 00 00 	lea    0x10ef(%rip),%rax        # 4043d5 <_IO_stdin_used+0x3d5>
  4032e6:	48 89 05 db 43 00 00 	mov    %rax,0x43db(%rip)        # 4076c8 <proc_info_features+0x5c8>
  4032ed:	48 8d 05 eb 10 00 00 	lea    0x10eb(%rip),%rax        # 4043df <_IO_stdin_used+0x3df>
  4032f4:	48 89 05 bd 43 00 00 	mov    %rax,0x43bd(%rip)        # 4076b8 <proc_info_features+0x5b8>
  4032fb:	c7 05 d3 43 00 00 43 	movl   $0x43,0x43d3(%rip)        # 4076d8 <proc_info_features+0x5d8>
  403302:	00 00 00 
  403305:	48 8d 05 dd 10 00 00 	lea    0x10dd(%rip),%rax        # 4043e9 <_IO_stdin_used+0x3e9>
  40330c:	48 89 05 cd 43 00 00 	mov    %rax,0x43cd(%rip)        # 4076e0 <proc_info_features+0x5e0>
  403313:	48 8d 05 d7 10 00 00 	lea    0x10d7(%rip),%rax        # 4043f1 <_IO_stdin_used+0x3f1>
  40331a:	48 89 05 af 43 00 00 	mov    %rax,0x43af(%rip)        # 4076d0 <proc_info_features+0x5d0>
  403321:	c7 05 c5 43 00 00 44 	movl   $0x44,0x43c5(%rip)        # 4076f0 <proc_info_features+0x5f0>
  403328:	00 00 00 
  40332b:	48 8d 05 c7 10 00 00 	lea    0x10c7(%rip),%rax        # 4043f9 <_IO_stdin_used+0x3f9>
  403332:	48 89 05 bf 43 00 00 	mov    %rax,0x43bf(%rip)        # 4076f8 <proc_info_features+0x5f8>
  403339:	48 8d 05 c4 10 00 00 	lea    0x10c4(%rip),%rax        # 404404 <_IO_stdin_used+0x404>
  403340:	48 89 05 a1 43 00 00 	mov    %rax,0x43a1(%rip)        # 4076e8 <proc_info_features+0x5e8>
  403347:	c7 05 b7 43 00 00 45 	movl   $0x45,0x43b7(%rip)        # 407708 <proc_info_features+0x608>
  40334e:	00 00 00 
  403351:	48 8d 05 b8 10 00 00 	lea    0x10b8(%rip),%rax        # 404410 <_IO_stdin_used+0x410>
  403358:	48 89 05 b1 43 00 00 	mov    %rax,0x43b1(%rip)        # 407710 <proc_info_features+0x610>
  40335f:	48 8d 05 b1 10 00 00 	lea    0x10b1(%rip),%rax        # 404417 <_IO_stdin_used+0x417>
  403366:	48 89 05 93 43 00 00 	mov    %rax,0x4393(%rip)        # 407700 <proc_info_features+0x600>
  40336d:	c7 05 a9 43 00 00 46 	movl   $0x46,0x43a9(%rip)        # 407720 <proc_info_features+0x620>
  403374:	00 00 00 
  403377:	48 8d 05 a0 10 00 00 	lea    0x10a0(%rip),%rax        # 40441e <_IO_stdin_used+0x41e>
  40337e:	48 89 05 a3 43 00 00 	mov    %rax,0x43a3(%rip)        # 407728 <proc_info_features+0x628>
  403385:	48 8d 05 9a 10 00 00 	lea    0x109a(%rip),%rax        # 404426 <_IO_stdin_used+0x426>
  40338c:	48 89 05 85 43 00 00 	mov    %rax,0x4385(%rip)        # 407718 <proc_info_features+0x618>
  403393:	c7 05 9b 43 00 00 47 	movl   $0x47,0x439b(%rip)        # 407738 <proc_info_features+0x638>
  40339a:	00 00 00 
  40339d:	48 8d 05 8b 10 00 00 	lea    0x108b(%rip),%rax        # 40442f <_IO_stdin_used+0x42f>
  4033a4:	48 89 05 95 43 00 00 	mov    %rax,0x4395(%rip)        # 407740 <proc_info_features+0x640>
  4033ab:	48 8d 05 86 10 00 00 	lea    0x1086(%rip),%rax        # 404438 <_IO_stdin_used+0x438>
  4033b2:	48 89 05 77 43 00 00 	mov    %rax,0x4377(%rip)        # 407730 <proc_info_features+0x630>
  4033b9:	c7 05 8d 43 00 00 48 	movl   $0x48,0x438d(%rip)        # 407750 <proc_info_features+0x650>
  4033c0:	00 00 00 
  4033c3:	48 8d 05 77 10 00 00 	lea    0x1077(%rip),%rax        # 404441 <_IO_stdin_used+0x441>
  4033ca:	48 89 05 87 43 00 00 	mov    %rax,0x4387(%rip)        # 407758 <proc_info_features+0x658>
  4033d1:	48 8d 05 72 10 00 00 	lea    0x1072(%rip),%rax        # 40444a <_IO_stdin_used+0x44a>
  4033d8:	48 89 05 69 43 00 00 	mov    %rax,0x4369(%rip)        # 407748 <proc_info_features+0x648>
  4033df:	c7 05 7f 43 00 00 49 	movl   $0x49,0x437f(%rip)        # 407768 <proc_info_features+0x668>
  4033e6:	00 00 00 
  4033e9:	48 8d 05 63 10 00 00 	lea    0x1063(%rip),%rax        # 404453 <_IO_stdin_used+0x453>
  4033f0:	48 89 05 79 43 00 00 	mov    %rax,0x4379(%rip)        # 407770 <proc_info_features+0x670>
  4033f7:	48 8d 05 5e 10 00 00 	lea    0x105e(%rip),%rax        # 40445c <_IO_stdin_used+0x45c>
  4033fe:	48 89 05 5b 43 00 00 	mov    %rax,0x435b(%rip)        # 407760 <proc_info_features+0x660>
  403405:	c7 05 71 43 00 00 4a 	movl   $0x4a,0x4371(%rip)        # 407780 <proc_info_features+0x680>
  40340c:	00 00 00 
  40340f:	48 8d 05 54 10 00 00 	lea    0x1054(%rip),%rax        # 40446a <_IO_stdin_used+0x46a>
  403416:	48 89 05 6b 43 00 00 	mov    %rax,0x436b(%rip)        # 407788 <proc_info_features+0x688>
  40341d:	48 8d 05 4e 10 00 00 	lea    0x104e(%rip),%rax        # 404472 <_IO_stdin_used+0x472>
  403424:	48 89 05 4d 43 00 00 	mov    %rax,0x434d(%rip)        # 407778 <proc_info_features+0x678>
  40342b:	c7 05 63 43 00 00 4b 	movl   $0x4b,0x4363(%rip)        # 407798 <proc_info_features+0x698>
  403432:	00 00 00 
  403435:	48 8d 05 29 10 00 00 	lea    0x1029(%rip),%rax        # 404465 <_IO_stdin_used+0x465>
  40343c:	48 89 05 5d 43 00 00 	mov    %rax,0x435d(%rip)        # 4077a0 <proc_info_features+0x6a0>
  403443:	48 8d 05 23 10 00 00 	lea    0x1023(%rip),%rax        # 40446d <_IO_stdin_used+0x46d>
  40344a:	48 89 05 3f 43 00 00 	mov    %rax,0x433f(%rip)        # 407790 <proc_info_features+0x690>
  403451:	c6 05 98 3c 00 00 01 	movb   $0x1,0x3c98(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  403458:	59                   	pop    %rcx
  403459:	c3                   	ret
  40345a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000403460 <__libirc_get_feature_bitpos>:
  403460:	f3 0f 1e fa          	endbr64
  403464:	51                   	push   %rcx
  403465:	89 c1                	mov    %eax,%ecx
  403467:	80 3d 82 3c 00 00 00 	cmpb   $0x0,0x3c82(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  40346e:	75 05                	jne    403475 <__libirc_get_feature_bitpos+0x15>
  403470:	e8 2b f5 ff ff       	call   4029a0 <__libirc_isa_init_once>
  403475:	89 c8                	mov    %ecx,%eax
  403477:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  40347b:	48 8d 0d 7e 3c 00 00 	lea    0x3c7e(%rip),%rcx        # 407100 <proc_info_features>
  403482:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  403486:	8d 41 80             	lea    -0x80(%rcx),%eax
  403489:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  40348e:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  403493:	0f 43 c1             	cmovae %ecx,%eax
  403496:	59                   	pop    %rcx
  403497:	c3                   	ret
  403498:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40349f:	00 

00000000004034a0 <__libirc_get_cpu_feature>:
  4034a0:	f3 0f 1e fa          	endbr64
  4034a4:	50                   	push   %rax
  4034a5:	80 3d 44 3c 00 00 00 	cmpb   $0x0,0x3c44(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  4034ac:	75 05                	jne    4034b3 <__libirc_get_cpu_feature+0x13>
  4034ae:	e8 ed f4 ff ff       	call   4029a0 <__libirc_isa_init_once>
  4034b3:	89 f0                	mov    %esi,%eax
  4034b5:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  4034b9:	48 8d 0d 40 3c 00 00 	lea    0x3c40(%rip),%rcx        # 407100 <proc_info_features>
  4034c0:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  4034c4:	8d 41 80             	lea    -0x80(%rcx),%eax
  4034c7:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  4034cc:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  4034d1:	0f 43 c1             	cmovae %ecx,%eax
  4034d4:	85 c0                	test   %eax,%eax
  4034d6:	78 14                	js     4034ec <__libirc_get_cpu_feature+0x4c>
  4034d8:	89 c1                	mov    %eax,%ecx
  4034da:	c1 e9 06             	shr    $0x6,%ecx
  4034dd:	48 8b 0c cf          	mov    (%rdi,%rcx,8),%rcx
  4034e1:	31 d2                	xor    %edx,%edx
  4034e3:	48 0f a3 c1          	bt     %rax,%rcx
  4034e7:	0f 92 c2             	setb   %dl
  4034ea:	89 d0                	mov    %edx,%eax
  4034ec:	59                   	pop    %rcx
  4034ed:	c3                   	ret
  4034ee:	66 90                	xchg   %ax,%ax

00000000004034f0 <__libirc_set_cpu_feature>:
  4034f0:	52                   	push   %rdx
  4034f1:	56                   	push   %rsi
  4034f2:	57                   	push   %rdi
  4034f3:	48 89 c2             	mov    %rax,%rdx
  4034f6:	80 3d f3 3b 00 00 00 	cmpb   $0x0,0x3bf3(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  4034fd:	75 05                	jne    403504 <__libirc_set_cpu_feature+0x14>
  4034ff:	e8 9c f4 ff ff       	call   4029a0 <__libirc_isa_init_once>
  403504:	89 c8                	mov    %ecx,%eax
  403506:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  40350a:	48 8d 0d ef 3b 00 00 	lea    0x3bef(%rip),%rcx        # 407100 <proc_info_features>
  403511:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  403515:	8d 41 80             	lea    -0x80(%rcx),%eax
  403518:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  40351d:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  403522:	0f 43 c1             	cmovae %ecx,%eax
  403525:	85 c0                	test   %eax,%eax
  403527:	78 18                	js     403541 <__libirc_set_cpu_feature+0x51>
  403529:	89 c6                	mov    %eax,%esi
  40352b:	c1 ee 06             	shr    $0x6,%esi
  40352e:	83 e0 3f             	and    $0x3f,%eax
  403531:	bf 01 00 00 00       	mov    $0x1,%edi
  403536:	89 c1                	mov    %eax,%ecx
  403538:	48 d3 e7             	shl    %cl,%rdi
  40353b:	48 09 3c f2          	or     %rdi,(%rdx,%rsi,8)
  40353f:	31 c0                	xor    %eax,%eax
  403541:	5f                   	pop    %rdi
  403542:	5e                   	pop    %rsi
  403543:	5a                   	pop    %rdx
  403544:	c3                   	ret
  403545:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  40354c:	00 00 00 
  40354f:	90                   	nop

0000000000403550 <__libirc_handle_intel_isa_disable>:
  403550:	55                   	push   %rbp
  403551:	41 57                	push   %r15
  403553:	41 56                	push   %r14
  403555:	41 54                	push   %r12
  403557:	53                   	push   %rbx
  403558:	31 db                	xor    %ebx,%ebx
  40355a:	48 85 ff             	test   %rdi,%rdi
  40355d:	0f 84 4b 01 00 00    	je     4036ae <__libirc_handle_intel_isa_disable+0x15e>
  403563:	49 89 fe             	mov    %rdi,%r14
  403566:	48 8d 3d 2a 0b 00 00 	lea    0xb2a(%rip),%rdi        # 404097 <_IO_stdin_used+0x97>
  40356d:	e8 be da ff ff       	call   401030 <getenv@plt>
  403572:	48 85 c0             	test   %rax,%rax
  403575:	0f 84 33 01 00 00    	je     4036ae <__libirc_handle_intel_isa_disable+0x15e>
  40357b:	48 89 c2             	mov    %rax,%rdx
  40357e:	0f b6 00             	movzbl (%rax),%eax
  403581:	84 c0                	test   %al,%al
  403583:	0f 84 25 01 00 00    	je     4036ae <__libirc_handle_intel_isa_disable+0x15e>
  403589:	31 db                	xor    %ebx,%ebx
  40358b:	48 8d 35 6e 3b 00 00 	lea    0x3b6e(%rip),%rsi        # 407100 <proc_info_features>
  403592:	31 ff                	xor    %edi,%edi
  403594:	4c 8d 42 01          	lea    0x1(%rdx),%r8
  403598:	41 b9 01 00 00 00    	mov    $0x1,%r9d
  40359e:	49 29 d1             	sub    %rdx,%r9
  4035a1:	49 89 d2             	mov    %rdx,%r10
  4035a4:	3c 2c                	cmp    $0x2c,%al
  4035a6:	75 1a                	jne    4035c2 <__libirc_handle_intel_isa_disable+0x72>
  4035a8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4035af:	00 
  4035b0:	41 0f b6 42 01       	movzbl 0x1(%r10),%eax
  4035b5:	49 ff c2             	inc    %r10
  4035b8:	49 ff c0             	inc    %r8
  4035bb:	49 ff c9             	dec    %r9
  4035be:	3c 2c                	cmp    $0x2c,%al
  4035c0:	74 ee                	je     4035b0 <__libirc_handle_intel_isa_disable+0x60>
  4035c2:	0f b6 c0             	movzbl %al,%eax
  4035c5:	85 c0                	test   %eax,%eax
  4035c7:	0f 84 e1 00 00 00    	je     4036ae <__libirc_handle_intel_isa_disable+0x15e>
  4035cd:	4c 89 c2             	mov    %r8,%rdx
  4035d0:	48 89 d0             	mov    %rdx,%rax
  4035d3:	0f b6 0a             	movzbl (%rdx),%ecx
  4035d6:	48 ff c2             	inc    %rdx
  4035d9:	83 f9 2c             	cmp    $0x2c,%ecx
  4035dc:	74 12                	je     4035f0 <__libirc_handle_intel_isa_disable+0xa0>
  4035de:	85 c9                	test   %ecx,%ecx
  4035e0:	74 0e                	je     4035f0 <__libirc_handle_intel_isa_disable+0xa0>
  4035e2:	48 89 c7             	mov    %rax,%rdi
  4035e5:	eb e9                	jmp    4035d0 <__libirc_handle_intel_isa_disable+0x80>
  4035e7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4035ee:	00 00 
  4035f0:	49 89 fb             	mov    %rdi,%r11
  4035f3:	4d 29 d3             	sub    %r10,%r11
  4035f6:	48 ff ca             	dec    %rdx
  4035f9:	49 ff c3             	inc    %r11
  4035fc:	75 0c                	jne    40360a <__libirc_handle_intel_isa_disable+0xba>
  4035fe:	0f b6 02             	movzbl (%rdx),%eax
  403601:	84 c0                	test   %al,%al
  403603:	75 8f                	jne    403594 <__libirc_handle_intel_isa_disable+0x44>
  403605:	e9 a4 00 00 00       	jmp    4036ae <__libirc_handle_intel_isa_disable+0x15e>
  40360a:	80 3d df 3a 00 00 00 	cmpb   $0x0,0x3adf(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  403611:	75 05                	jne    403618 <__libirc_handle_intel_isa_disable+0xc8>
  403613:	e8 88 f3 ff ff       	call   4029a0 <__libirc_isa_init_once>
  403618:	4c 89 d8             	mov    %r11,%rax
  40361b:	48 83 e0 fe          	and    $0xfffffffffffffffe,%rax
  40361f:	49 01 f9             	add    %rdi,%r9
  403622:	49 d1 e9             	shr    %r9
  403625:	b9 01 00 00 00       	mov    $0x1,%ecx
  40362a:	eb 14                	jmp    403640 <__libirc_handle_intel_isa_disable+0xf0>
  40362c:	0f 1f 40 00          	nopl   0x0(%rax)
  403630:	43 80 3c 1f 00       	cmpb   $0x0,(%r15,%r11,1)
  403635:	74 5b                	je     403692 <__libirc_handle_intel_isa_disable+0x142>
  403637:	48 ff c1             	inc    %rcx
  40363a:	48 83 f9 47          	cmp    $0x47,%rcx
  40363e:	74 be                	je     4035fe <__libirc_handle_intel_isa_disable+0xae>
  403640:	4c 8d 3c 49          	lea    (%rcx,%rcx,2),%r15
  403644:	4e 8b 7c fe 10       	mov    0x10(%rsi,%r15,8),%r15
  403649:	4d 85 ff             	test   %r15,%r15
  40364c:	74 e9                	je     403637 <__libirc_handle_intel_isa_disable+0xe7>
  40364e:	49 83 fb 02          	cmp    $0x2,%r11
  403652:	72 2c                	jb     403680 <__libirc_handle_intel_isa_disable+0x130>
  403654:	45 31 e4             	xor    %r12d,%r12d
  403657:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40365e:	00 00 
  403660:	43 0f b6 6c 60 ff    	movzbl -0x1(%r8,%r12,2),%ebp
  403666:	43 3a 2c 67          	cmp    (%r15,%r12,2),%bpl
  40366a:	75 cb                	jne    403637 <__libirc_handle_intel_isa_disable+0xe7>
  40366c:	43 0f b6 2c 60       	movzbl (%r8,%r12,2),%ebp
  403671:	43 3a 6c 67 01       	cmp    0x1(%r15,%r12,2),%bpl
  403676:	75 bf                	jne    403637 <__libirc_handle_intel_isa_disable+0xe7>
  403678:	49 ff c4             	inc    %r12
  40367b:	4d 39 e1             	cmp    %r12,%r9
  40367e:	75 e0                	jne    403660 <__libirc_handle_intel_isa_disable+0x110>
  403680:	4c 39 d8             	cmp    %r11,%rax
  403683:	73 ab                	jae    403630 <__libirc_handle_intel_isa_disable+0xe0>
  403685:	41 0f b6 2c 02       	movzbl (%r10,%rax,1),%ebp
  40368a:	41 3a 2c 07          	cmp    (%r15,%rax,1),%bpl
  40368e:	74 a0                	je     403630 <__libirc_handle_intel_isa_disable+0xe0>
  403690:	eb a5                	jmp    403637 <__libirc_handle_intel_isa_disable+0xe7>
  403692:	83 f9 02             	cmp    $0x2,%ecx
  403695:	0f 82 63 ff ff ff    	jb     4035fe <__libirc_handle_intel_isa_disable+0xae>
  40369b:	4c 89 f0             	mov    %r14,%rax
  40369e:	e8 4d fe ff ff       	call   4034f0 <__libirc_set_cpu_feature>
  4036a3:	83 f8 01             	cmp    $0x1,%eax
  4036a6:	83 d3 00             	adc    $0x0,%ebx
  4036a9:	e9 50 ff ff ff       	jmp    4035fe <__libirc_handle_intel_isa_disable+0xae>
  4036ae:	89 d8                	mov    %ebx,%eax
  4036b0:	5b                   	pop    %rbx
  4036b1:	41 5c                	pop    %r12
  4036b3:	41 5e                	pop    %r14
  4036b5:	41 5f                	pop    %r15
  4036b7:	5d                   	pop    %rbp
  4036b8:	c3                   	ret
  4036b9:	0f 1f 00             	nopl   (%rax)
  4036bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004036c0 <__libirc_get_msg>:
  4036c0:	f3 0f 1e fa          	endbr64
  4036c4:	53                   	push   %rbx
  4036c5:	48 81 ec d0 00 00 00 	sub    $0xd0,%rsp
  4036cc:	89 f3                	mov    %esi,%ebx
  4036ce:	48 89 54 24 30       	mov    %rdx,0x30(%rsp)
  4036d3:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
  4036d8:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
  4036dd:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
  4036e2:	84 c0                	test   %al,%al
  4036e4:	74 37                	je     40371d <__libirc_get_msg+0x5d>
  4036e6:	0f 29 44 24 50       	movaps %xmm0,0x50(%rsp)
  4036eb:	0f 29 4c 24 60       	movaps %xmm1,0x60(%rsp)
  4036f0:	0f 29 54 24 70       	movaps %xmm2,0x70(%rsp)
  4036f5:	0f 29 9c 24 80 00 00 	movaps %xmm3,0x80(%rsp)
  4036fc:	00 
  4036fd:	0f 29 a4 24 90 00 00 	movaps %xmm4,0x90(%rsp)
  403704:	00 
  403705:	0f 29 ac 24 a0 00 00 	movaps %xmm5,0xa0(%rsp)
  40370c:	00 
  40370d:	0f 29 b4 24 b0 00 00 	movaps %xmm6,0xb0(%rsp)
  403714:	00 
  403715:	0f 29 bc 24 c0 00 00 	movaps %xmm7,0xc0(%rsp)
  40371c:	00 
  40371d:	e8 5e 00 00 00       	call   403780 <irc_ptr_msg>
  403722:	85 db                	test   %ebx,%ebx
  403724:	7e 4c                	jle    403772 <__libirc_get_msg+0xb2>
  403726:	48 8d 4c 24 20       	lea    0x20(%rsp),%rcx
  40372b:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  403730:	48 8d 8c 24 e0 00 00 	lea    0xe0(%rsp),%rcx
  403737:	00 
  403738:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  40373d:	48 b9 10 00 00 00 30 	movabs $0x3000000010,%rcx
  403744:	00 00 00 
  403747:	48 89 0c 24          	mov    %rcx,(%rsp)
  40374b:	48 8d 1d 5e 40 00 00 	lea    0x405e(%rip),%rbx        # 4077b0 <get_msg_buf>
  403752:	49 89 e1             	mov    %rsp,%r9
  403755:	be 00 02 00 00       	mov    $0x200,%esi
  40375a:	b9 00 02 00 00       	mov    $0x200,%ecx
  40375f:	48 89 df             	mov    %rbx,%rdi
  403762:	ba 01 00 00 00       	mov    $0x1,%edx
  403767:	49 89 c0             	mov    %rax,%r8
  40376a:	e8 91 d9 ff ff       	call   401100 <__vsnprintf_chk@plt>
  40376f:	48 89 d8             	mov    %rbx,%rax
  403772:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  403779:	5b                   	pop    %rbx
  40377a:	c3                   	ret
  40377b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000403780 <irc_ptr_msg>:
  403780:	41 57                	push   %r15
  403782:	41 56                	push   %r14
  403784:	41 54                	push   %r12
  403786:	53                   	push   %rbx
  403787:	48 81 ec 88 00 00 00 	sub    $0x88,%rsp
  40378e:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  403795:	00 00 
  403797:	48 89 84 24 80 00 00 	mov    %rax,0x80(%rsp)
  40379e:	00 
  40379f:	85 ff                	test   %edi,%edi
  4037a1:	74 37                	je     4037da <irc_ptr_msg+0x5a>
  4037a3:	89 fb                	mov    %edi,%ebx
  4037a5:	80 3d 04 44 00 00 00 	cmpb   $0x0,0x4404(%rip)        # 407bb0 <first_msg>
  4037ac:	74 38                	je     4037e6 <irc_ptr_msg+0x66>
  4037ae:	48 63 c3             	movslq %ebx,%rax
  4037b1:	48 c1 e0 04          	shl    $0x4,%rax
  4037b5:	48 8d 0d 14 32 00 00 	lea    0x3214(%rip),%rcx        # 4069d0 <irc_msgtab>
  4037bc:	48 8b 44 08 08       	mov    0x8(%rax,%rcx,1),%rax
  4037c1:	80 3d ec 43 00 00 01 	cmpb   $0x1,0x43ec(%rip)        # 407bb4 <use_internal_msg>
  4037c8:	0f 85 04 01 00 00    	jne    4038d2 <irc_ptr_msg+0x152>
  4037ce:	48 8b 3d e3 43 00 00 	mov    0x43e3(%rip),%rdi        # 407bb8 <message_catalog>
  4037d5:	e9 e9 00 00 00       	jmp    4038c3 <irc_ptr_msg+0x143>
  4037da:	48 8d 05 97 0c 00 00 	lea    0xc97(%rip),%rax        # 404478 <_IO_stdin_used+0x478>
  4037e1:	e9 ec 00 00 00       	jmp    4038d2 <irc_ptr_msg+0x152>
  4037e6:	c6 05 c3 43 00 00 01 	movb   $0x1,0x43c3(%rip)        # 407bb0 <first_msg>
  4037ed:	48 8d 3d 85 0c 00 00 	lea    0xc85(%rip),%rdi        # 404479 <_IO_stdin_used+0x479>
  4037f4:	31 f6                	xor    %esi,%esi
  4037f6:	e8 f5 d8 ff ff       	call   4010f0 <catopen@plt>
  4037fb:	48 89 c7             	mov    %rax,%rdi
  4037fe:	48 89 05 b3 43 00 00 	mov    %rax,0x43b3(%rip)        # 407bb8 <message_catalog>
  403805:	48 83 f8 ff          	cmp    $0xffffffffffffffff,%rax
  403809:	0f 85 9a 00 00 00    	jne    4038a9 <irc_ptr_msg+0x129>
  40380f:	48 8d 3d 6f 0c 00 00 	lea    0xc6f(%rip),%rdi        # 404485 <_IO_stdin_used+0x485>
  403816:	e8 15 d8 ff ff       	call   401030 <getenv@plt>
  40381b:	48 85 c0             	test   %rax,%rax
  40381e:	74 78                	je     403898 <irc_ptr_msg+0x118>
  403820:	49 89 e6             	mov    %rsp,%r14
  403823:	ba 80 00 00 00       	mov    $0x80,%edx
  403828:	b9 80 00 00 00       	mov    $0x80,%ecx
  40382d:	4c 89 f7             	mov    %r14,%rdi
  403830:	48 89 c6             	mov    %rax,%rsi
  403833:	e8 d8 d8 ff ff       	call   401110 <__strncpy_chk@plt>
  403838:	c6 44 24 7f 00       	movb   $0x0,0x7f(%rsp)
  40383d:	4c 89 f7             	mov    %r14,%rdi
  403840:	be 2e 00 00 00       	mov    $0x2e,%esi
  403845:	e8 56 d8 ff ff       	call   4010a0 <strchr@plt>
  40384a:	48 85 c0             	test   %rax,%rax
  40384d:	74 49                	je     403898 <irc_ptr_msg+0x118>
  40384f:	49 89 c6             	mov    %rax,%r14
  403852:	c6 00 00             	movb   $0x0,(%rax)
  403855:	4c 8d 3d 29 0c 00 00 	lea    0xc29(%rip),%r15        # 404485 <_IO_stdin_used+0x485>
  40385c:	49 89 e4             	mov    %rsp,%r12
  40385f:	4c 89 ff             	mov    %r15,%rdi
  403862:	4c 89 e6             	mov    %r12,%rsi
  403865:	ba 01 00 00 00       	mov    $0x1,%edx
  40386a:	e8 e1 d7 ff ff       	call   401050 <setenv@plt>
  40386f:	48 8d 3d 03 0c 00 00 	lea    0xc03(%rip),%rdi        # 404479 <_IO_stdin_used+0x479>
  403876:	31 f6                	xor    %esi,%esi
  403878:	e8 73 d8 ff ff       	call   4010f0 <catopen@plt>
  40387d:	48 89 05 34 43 00 00 	mov    %rax,0x4334(%rip)        # 407bb8 <message_catalog>
  403884:	41 c6 06 2e          	movb   $0x2e,(%r14)
  403888:	4c 89 ff             	mov    %r15,%rdi
  40388b:	4c 89 e6             	mov    %r12,%rsi
  40388e:	ba 01 00 00 00       	mov    $0x1,%edx
  403893:	e8 b8 d7 ff ff       	call   401050 <setenv@plt>
  403898:	48 8b 3d 19 43 00 00 	mov    0x4319(%rip),%rdi        # 407bb8 <message_catalog>
  40389f:	48 83 ff ff          	cmp    $0xffffffffffffffff,%rdi
  4038a3:	0f 84 05 ff ff ff    	je     4037ae <irc_ptr_msg+0x2e>
  4038a9:	c6 05 04 43 00 00 01 	movb   $0x1,0x4304(%rip)        # 407bb4 <use_internal_msg>
  4038b0:	48 63 c3             	movslq %ebx,%rax
  4038b3:	48 c1 e0 04          	shl    $0x4,%rax
  4038b7:	48 8d 0d 12 31 00 00 	lea    0x3112(%rip),%rcx        # 4069d0 <irc_msgtab>
  4038be:	48 8b 44 08 08       	mov    0x8(%rax,%rcx,1),%rax
  4038c3:	be 01 00 00 00       	mov    $0x1,%esi
  4038c8:	89 da                	mov    %ebx,%edx
  4038ca:	48 89 c1             	mov    %rax,%rcx
  4038cd:	e8 7e d8 ff ff       	call   401150 <catgets@plt>
  4038d2:	64 48 8b 0c 25 28 00 	mov    %fs:0x28,%rcx
  4038d9:	00 00 
  4038db:	48 3b 8c 24 80 00 00 	cmp    0x80(%rsp),%rcx
  4038e2:	00 
  4038e3:	75 0f                	jne    4038f4 <irc_ptr_msg+0x174>
  4038e5:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
  4038ec:	5b                   	pop    %rbx
  4038ed:	41 5c                	pop    %r12
  4038ef:	41 5e                	pop    %r14
  4038f1:	41 5f                	pop    %r15
  4038f3:	c3                   	ret
  4038f4:	e8 97 d7 ff ff       	call   401090 <__stack_chk_fail@plt>
  4038f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000403900 <__libirc_print>:
  403900:	f3 0f 1e fa          	endbr64
  403904:	55                   	push   %rbp
  403905:	41 56                	push   %r14
  403907:	53                   	push   %rbx
  403908:	48 81 ec d0 00 00 00 	sub    $0xd0,%rsp
  40390f:	89 fb                	mov    %edi,%ebx
  403911:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
  403916:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
  40391b:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
  403920:	84 c0                	test   %al,%al
  403922:	74 37                	je     40395b <__libirc_print+0x5b>
  403924:	0f 29 44 24 50       	movaps %xmm0,0x50(%rsp)
  403929:	0f 29 4c 24 60       	movaps %xmm1,0x60(%rsp)
  40392e:	0f 29 54 24 70       	movaps %xmm2,0x70(%rsp)
  403933:	0f 29 9c 24 80 00 00 	movaps %xmm3,0x80(%rsp)
  40393a:	00 
  40393b:	0f 29 a4 24 90 00 00 	movaps %xmm4,0x90(%rsp)
  403942:	00 
  403943:	0f 29 ac 24 a0 00 00 	movaps %xmm5,0xa0(%rsp)
  40394a:	00 
  40394b:	0f 29 b4 24 b0 00 00 	movaps %xmm6,0xb0(%rsp)
  403952:	00 
  403953:	0f 29 bc 24 c0 00 00 	movaps %xmm7,0xc0(%rsp)
  40395a:	00 
  40395b:	85 f6                	test   %esi,%esi
  40395d:	0f 84 80 00 00 00    	je     4039e3 <__libirc_print+0xe3>
  403963:	89 d5                	mov    %edx,%ebp
  403965:	89 f7                	mov    %esi,%edi
  403967:	e8 14 fe ff ff       	call   403780 <irc_ptr_msg>
  40396c:	85 ed                	test   %ebp,%ebp
  40396e:	7e 4c                	jle    4039bc <__libirc_print+0xbc>
  403970:	48 8d 4c 24 20       	lea    0x20(%rsp),%rcx
  403975:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  40397a:	48 8d 8c 24 f0 00 00 	lea    0xf0(%rsp),%rcx
  403981:	00 
  403982:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  403987:	48 b9 18 00 00 00 30 	movabs $0x3000000018,%rcx
  40398e:	00 00 00 
  403991:	48 89 0c 24          	mov    %rcx,(%rsp)
  403995:	4c 8d 35 14 40 00 00 	lea    0x4014(%rip),%r14        # 4079b0 <print_buf>
  40399c:	49 89 e1             	mov    %rsp,%r9
  40399f:	be 00 02 00 00       	mov    $0x200,%esi
  4039a4:	b9 00 02 00 00       	mov    $0x200,%ecx
  4039a9:	4c 89 f7             	mov    %r14,%rdi
  4039ac:	ba 01 00 00 00       	mov    $0x1,%edx
  4039b1:	49 89 c0             	mov    %rax,%r8
  4039b4:	e8 47 d7 ff ff       	call   401100 <__vsnprintf_chk@plt>
  4039b9:	4c 89 f0             	mov    %r14,%rax
  4039bc:	83 fb 01             	cmp    $0x1,%ebx
  4039bf:	75 4f                	jne    403a10 <__libirc_print+0x110>
  4039c1:	48 8b 0d 10 36 00 00 	mov    0x3610(%rip),%rcx        # 406fd8 <stderr@GLIBC_2.2.5-0xe8>
  4039c8:	48 8b 39             	mov    (%rcx),%rdi
  4039cb:	48 8d 15 a3 0a 00 00 	lea    0xaa3(%rip),%rdx        # 404475 <_IO_stdin_used+0x475>
  4039d2:	be 01 00 00 00       	mov    $0x1,%esi
  4039d7:	48 89 c1             	mov    %rax,%rcx
  4039da:	31 c0                	xor    %eax,%eax
  4039dc:	e8 8f d7 ff ff       	call   401170 <__fprintf_chk@plt>
  4039e1:	eb 43                	jmp    403a26 <__libirc_print+0x126>
  4039e3:	83 fb 01             	cmp    $0x1,%ebx
  4039e6:	75 4a                	jne    403a32 <__libirc_print+0x132>
  4039e8:	48 8b 05 e9 35 00 00 	mov    0x35e9(%rip),%rax        # 406fd8 <stderr@GLIBC_2.2.5-0xe8>
  4039ef:	48 8b 38             	mov    (%rax),%rdi
  4039f2:	48 8d 15 7e 0a 00 00 	lea    0xa7e(%rip),%rdx        # 404477 <_IO_stdin_used+0x477>
  4039f9:	be 01 00 00 00       	mov    $0x1,%esi
  4039fe:	31 c0                	xor    %eax,%eax
  403a00:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  403a07:	5b                   	pop    %rbx
  403a08:	41 5e                	pop    %r14
  403a0a:	5d                   	pop    %rbp
  403a0b:	e9 60 d7 ff ff       	jmp    401170 <__fprintf_chk@plt>
  403a10:	48 8d 35 5e 0a 00 00 	lea    0xa5e(%rip),%rsi        # 404475 <_IO_stdin_used+0x475>
  403a17:	bf 01 00 00 00       	mov    $0x1,%edi
  403a1c:	48 89 c2             	mov    %rax,%rdx
  403a1f:	31 c0                	xor    %eax,%eax
  403a21:	e8 fa d6 ff ff       	call   401120 <__printf_chk@plt>
  403a26:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  403a2d:	5b                   	pop    %rbx
  403a2e:	41 5e                	pop    %r14
  403a30:	5d                   	pop    %rbp
  403a31:	c3                   	ret
  403a32:	48 8d 35 3e 0a 00 00 	lea    0xa3e(%rip),%rsi        # 404477 <_IO_stdin_used+0x477>
  403a39:	bf 01 00 00 00       	mov    $0x1,%edi
  403a3e:	31 c0                	xor    %eax,%eax
  403a40:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  403a47:	5b                   	pop    %rbx
  403a48:	41 5e                	pop    %r14
  403a4a:	5d                   	pop    %rbp
  403a4b:	e9 d0 d6 ff ff       	jmp    401120 <__printf_chk@plt>

Disassembly of section .fini:

0000000000403a50 <_fini>:
  403a50:	48 83 ec 08          	sub    $0x8,%rsp
  403a54:	48 83 c4 08          	add    $0x8,%rsp
  403a58:	c3                   	ret
