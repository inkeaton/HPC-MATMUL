
bin/seq_bench/icx/matmul_O3_xHost_unroll:     file format elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
  401000:	48 83 ec 08          	sub    $0x8,%rsp
  401004:	48 8b 05 b5 6f 00 00 	mov    0x6fb5(%rip),%rax        # 407fc0 <__gmon_start__@Base>
  40100b:	48 85 c0             	test   %rax,%rax
  40100e:	74 02                	je     401012 <_init+0x12>
  401010:	ff d0                	call   *%rax
  401012:	48 83 c4 08          	add    $0x8,%rsp
  401016:	c3                   	ret

Disassembly of section .plt:

0000000000401020 <getenv@plt-0x10>:
  401020:	ff 35 ca 6f 00 00    	push   0x6fca(%rip)        # 407ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
  401026:	ff 25 cc 6f 00 00    	jmp    *0x6fcc(%rip)        # 407ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
  40102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401030 <getenv@plt>:
  401030:	ff 25 ca 6f 00 00    	jmp    *0x6fca(%rip)        # 408000 <getenv@GLIBC_2.2.5>
  401036:	68 00 00 00 00       	push   $0x0
  40103b:	e9 e0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401040 <free@plt>:
  401040:	ff 25 c2 6f 00 00    	jmp    *0x6fc2(%rip)        # 408008 <free@GLIBC_2.2.5>
  401046:	68 01 00 00 00       	push   $0x1
  40104b:	e9 d0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401050 <setenv@plt>:
  401050:	ff 25 ba 6f 00 00    	jmp    *0x6fba(%rip)        # 408010 <setenv@GLIBC_2.2.5>
  401056:	68 02 00 00 00       	push   $0x2
  40105b:	e9 c0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401060 <clock@plt>:
  401060:	ff 25 b2 6f 00 00    	jmp    *0x6fb2(%rip)        # 408018 <clock@GLIBC_2.2.5>
  401066:	68 03 00 00 00       	push   $0x3
  40106b:	e9 b0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401070 <fclose@plt>:
  401070:	ff 25 aa 6f 00 00    	jmp    *0x6faa(%rip)        # 408020 <fclose@GLIBC_2.2.5>
  401076:	68 04 00 00 00       	push   $0x4
  40107b:	e9 a0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401080 <strlen@plt>:
  401080:	ff 25 a2 6f 00 00    	jmp    *0x6fa2(%rip)        # 408028 <strlen@GLIBC_2.2.5>
  401086:	68 05 00 00 00       	push   $0x5
  40108b:	e9 90 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401090 <__stack_chk_fail@plt>:
  401090:	ff 25 9a 6f 00 00    	jmp    *0x6f9a(%rip)        # 408030 <__stack_chk_fail@GLIBC_2.4>
  401096:	68 06 00 00 00       	push   $0x6
  40109b:	e9 80 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010a0 <strchr@plt>:
  4010a0:	ff 25 92 6f 00 00    	jmp    *0x6f92(%rip)        # 408038 <strchr@GLIBC_2.2.5>
  4010a6:	68 07 00 00 00       	push   $0x7
  4010ab:	e9 70 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010b0 <memset@plt>:
  4010b0:	ff 25 8a 6f 00 00    	jmp    *0x6f8a(%rip)        # 408040 <memset@GLIBC_2.2.5>
  4010b6:	68 08 00 00 00       	push   $0x8
  4010bb:	e9 60 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010c0 <fputc@plt>:
  4010c0:	ff 25 82 6f 00 00    	jmp    *0x6f82(%rip)        # 408048 <fputc@GLIBC_2.2.5>
  4010c6:	68 09 00 00 00       	push   $0x9
  4010cb:	e9 50 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010d0 <calloc@plt>:
  4010d0:	ff 25 7a 6f 00 00    	jmp    *0x6f7a(%rip)        # 408050 <calloc@GLIBC_2.2.5>
  4010d6:	68 0a 00 00 00       	push   $0xa
  4010db:	e9 40 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010e0 <fprintf@plt>:
  4010e0:	ff 25 72 6f 00 00    	jmp    *0x6f72(%rip)        # 408058 <fprintf@GLIBC_2.2.5>
  4010e6:	68 0b 00 00 00       	push   $0xb
  4010eb:	e9 30 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010f0 <memcpy@plt>:
  4010f0:	ff 25 6a 6f 00 00    	jmp    *0x6f6a(%rip)        # 408060 <memcpy@GLIBC_2.14>
  4010f6:	68 0c 00 00 00       	push   $0xc
  4010fb:	e9 20 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401100 <malloc@plt>:
  401100:	ff 25 62 6f 00 00    	jmp    *0x6f62(%rip)        # 408068 <malloc@GLIBC_2.2.5>
  401106:	68 0d 00 00 00       	push   $0xd
  40110b:	e9 10 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401110 <catopen@plt>:
  401110:	ff 25 5a 6f 00 00    	jmp    *0x6f5a(%rip)        # 408070 <catopen@GLIBC_2.2.5>
  401116:	68 0e 00 00 00       	push   $0xe
  40111b:	e9 00 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401120 <__vsnprintf_chk@plt>:
  401120:	ff 25 52 6f 00 00    	jmp    *0x6f52(%rip)        # 408078 <__vsnprintf_chk@GLIBC_2.3.4>
  401126:	68 0f 00 00 00       	push   $0xf
  40112b:	e9 f0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401130 <__strncpy_chk@plt>:
  401130:	ff 25 4a 6f 00 00    	jmp    *0x6f4a(%rip)        # 408080 <__strncpy_chk@GLIBC_2.3.4>
  401136:	68 10 00 00 00       	push   $0x10
  40113b:	e9 e0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401140 <__printf_chk@plt>:
  401140:	ff 25 42 6f 00 00    	jmp    *0x6f42(%rip)        # 408088 <__printf_chk@GLIBC_2.3.4>
  401146:	68 11 00 00 00       	push   $0x11
  40114b:	e9 d0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401150 <fopen@plt>:
  401150:	ff 25 3a 6f 00 00    	jmp    *0x6f3a(%rip)        # 408090 <fopen@GLIBC_2.2.5>
  401156:	68 12 00 00 00       	push   $0x12
  40115b:	e9 c0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401160 <perror@plt>:
  401160:	ff 25 32 6f 00 00    	jmp    *0x6f32(%rip)        # 408098 <perror@GLIBC_2.2.5>
  401166:	68 13 00 00 00       	push   $0x13
  40116b:	e9 b0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401170 <catgets@plt>:
  401170:	ff 25 2a 6f 00 00    	jmp    *0x6f2a(%rip)        # 4080a0 <catgets@GLIBC_2.2.5>
  401176:	68 14 00 00 00       	push   $0x14
  40117b:	e9 a0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401180 <exit@plt>:
  401180:	ff 25 22 6f 00 00    	jmp    *0x6f22(%rip)        # 4080a8 <exit@GLIBC_2.2.5>
  401186:	68 15 00 00 00       	push   $0x15
  40118b:	e9 90 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401190 <__fprintf_chk@plt>:
  401190:	ff 25 1a 6f 00 00    	jmp    *0x6f1a(%rip)        # 4080b0 <__fprintf_chk@GLIBC_2.3.4>
  401196:	68 16 00 00 00       	push   $0x16
  40119b:	e9 80 fe ff ff       	jmp    401020 <_init+0x20>

00000000004011a0 <__strncat_chk@plt>:
  4011a0:	ff 25 12 6f 00 00    	jmp    *0x6f12(%rip)        # 4080b8 <__strncat_chk@GLIBC_2.3.4>
  4011a6:	68 17 00 00 00       	push   $0x17
  4011ab:	e9 70 fe ff ff       	jmp    401020 <_init+0x20>

Disassembly of section .plt.got:

00000000004011b0 <__cxa_finalize@plt>:
  4011b0:	ff 25 1a 6e 00 00    	jmp    *0x6e1a(%rip)        # 407fd0 <__cxa_finalize@GLIBC_2.2.5>
  4011b6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004011c0 <_start>:
  4011c0:	31 ed                	xor    %ebp,%ebp
  4011c2:	49 89 d1             	mov    %rdx,%r9
  4011c5:	5e                   	pop    %rsi
  4011c6:	48 89 e2             	mov    %rsp,%rdx
  4011c9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4011cd:	50                   	push   %rax
  4011ce:	54                   	push   %rsp
  4011cf:	45 31 c0             	xor    %r8d,%r8d
  4011d2:	31 c9                	xor    %ecx,%ecx
  4011d4:	48 c7 c7 b0 12 40 00 	mov    $0x4012b0,%rdi
  4011db:	ff 15 cf 6d 00 00    	call   *0x6dcf(%rip)        # 407fb0 <__libc_start_main@GLIBC_2.34>
  4011e1:	f4                   	hlt
  4011e2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4011e9:	00 00 00 
  4011ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004011f0 <deregister_tm_clones>:
  4011f0:	48 8d 3d d9 6e 00 00 	lea    0x6ed9(%rip),%rdi        # 4080d0 <__TMC_END__>
  4011f7:	48 8d 05 d2 6e 00 00 	lea    0x6ed2(%rip),%rax        # 4080d0 <__TMC_END__>
  4011fe:	48 39 f8             	cmp    %rdi,%rax
  401201:	74 15                	je     401218 <deregister_tm_clones+0x28>
  401203:	48 8b 05 ae 6d 00 00 	mov    0x6dae(%rip),%rax        # 407fb8 <_ITM_deregisterTMCloneTable@Base>
  40120a:	48 85 c0             	test   %rax,%rax
  40120d:	74 09                	je     401218 <deregister_tm_clones+0x28>
  40120f:	ff e0                	jmp    *%rax
  401211:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401218:	c3                   	ret
  401219:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401220 <register_tm_clones>:
  401220:	48 8d 3d a9 6e 00 00 	lea    0x6ea9(%rip),%rdi        # 4080d0 <__TMC_END__>
  401227:	48 8d 35 a2 6e 00 00 	lea    0x6ea2(%rip),%rsi        # 4080d0 <__TMC_END__>
  40122e:	48 29 fe             	sub    %rdi,%rsi
  401231:	48 89 f0             	mov    %rsi,%rax
  401234:	48 c1 ee 3f          	shr    $0x3f,%rsi
  401238:	48 c1 f8 03          	sar    $0x3,%rax
  40123c:	48 01 c6             	add    %rax,%rsi
  40123f:	48 d1 fe             	sar    %rsi
  401242:	74 14                	je     401258 <register_tm_clones+0x38>
  401244:	48 8b 05 7d 6d 00 00 	mov    0x6d7d(%rip),%rax        # 407fc8 <_ITM_registerTMCloneTable@Base>
  40124b:	48 85 c0             	test   %rax,%rax
  40124e:	74 08                	je     401258 <register_tm_clones+0x38>
  401250:	ff e0                	jmp    *%rax
  401252:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401258:	c3                   	ret
  401259:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401260 <__do_global_dtors_aux>:
  401260:	f3 0f 1e fa          	endbr64
  401264:	80 3d 7d 6e 00 00 00 	cmpb   $0x0,0x6e7d(%rip)        # 4080e8 <completed.0>
  40126b:	75 2b                	jne    401298 <__do_global_dtors_aux+0x38>
  40126d:	55                   	push   %rbp
  40126e:	48 83 3d 5a 6d 00 00 	cmpq   $0x0,0x6d5a(%rip)        # 407fd0 <__cxa_finalize@GLIBC_2.2.5>
  401275:	00 
  401276:	48 89 e5             	mov    %rsp,%rbp
  401279:	74 0c                	je     401287 <__do_global_dtors_aux+0x27>
  40127b:	48 8b 3d 46 6e 00 00 	mov    0x6e46(%rip),%rdi        # 4080c8 <__dso_handle>
  401282:	e8 29 ff ff ff       	call   4011b0 <__cxa_finalize@plt>
  401287:	e8 64 ff ff ff       	call   4011f0 <deregister_tm_clones>
  40128c:	c6 05 55 6e 00 00 01 	movb   $0x1,0x6e55(%rip)        # 4080e8 <completed.0>
  401293:	5d                   	pop    %rbp
  401294:	c3                   	ret
  401295:	0f 1f 00             	nopl   (%rax)
  401298:	c3                   	ret
  401299:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004012a0 <frame_dummy>:
  4012a0:	f3 0f 1e fa          	endbr64
  4012a4:	e9 77 ff ff ff       	jmp    401220 <register_tm_clones>
  4012a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004012b0 <main>:
  4012b0:	55                   	push   %rbp
  4012b1:	48 89 e5             	mov    %rsp,%rbp
  4012b4:	41 57                	push   %r15
  4012b6:	41 56                	push   %r14
  4012b8:	41 55                	push   %r13
  4012ba:	41 54                	push   %r12
  4012bc:	53                   	push   %rbx
  4012bd:	48 83 e4 c0          	and    $0xffffffffffffffc0,%rsp
  4012c1:	48 81 ec c0 22 00 00 	sub    $0x22c0,%rsp
  4012c8:	48 be ee 9f 9d 10 00 	movabs $0x100000109d9fee,%rsi
  4012cf:	00 10 00 
  4012d2:	bf 03 00 00 00       	mov    $0x3,%edi
  4012d7:	e8 94 10 00 00       	call   402370 <__intel_new_feature_proc_init>
  4012dc:	4c 8d ac 24 58 01 00 	lea    0x158(%rsp),%r13
  4012e3:	00 
  4012e4:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4012e9:	e8 12 fe ff ff       	call   401100 <malloc@plt>
  4012ee:	48 89 c3             	mov    %rax,%rbx
  4012f1:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4012f6:	e8 05 fe ff ff       	call   401100 <malloc@plt>
  4012fb:	49 89 c6             	mov    %rax,%r14
  4012fe:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  401303:	bf 01 00 00 00       	mov    $0x1,%edi
  401308:	be 00 c2 eb 0b       	mov    $0xbebc200,%esi
  40130d:	e8 be fd ff ff       	call   4010d0 <calloc@plt>
  401312:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  401317:	48 89 9c 24 00 12 00 	mov    %rbx,0x1200(%rsp)
  40131e:	00 
  40131f:	48 89 5c 24 28       	mov    %rbx,0x28(%rsp)
  401324:	89 d8                	mov    %ebx,%eax
  401326:	83 e0 3f             	and    $0x3f,%eax
  401329:	48 89 84 24 08 12 00 	mov    %rax,0x1208(%rsp)
  401330:	00 
  401331:	48 c7 84 24 10 12 00 	movq   $0x0,0x1210(%rsp)
  401338:	00 00 00 00 00 
  40133d:	4c 89 b4 24 40 01 00 	mov    %r14,0x140(%rsp)
  401344:	00 
  401345:	4c 89 74 24 10       	mov    %r14,0x10(%rsp)
  40134a:	44 89 f0             	mov    %r14d,%eax
  40134d:	83 e0 3f             	and    $0x3f,%eax
  401350:	48 89 84 24 48 01 00 	mov    %rax,0x148(%rsp)
  401357:	00 
  401358:	48 c7 84 24 50 01 00 	movq   $0x0,0x150(%rsp)
  40135f:	00 00 00 00 00 
  401364:	49 c7 c6 d8 ff ff ff 	mov    $0xffffffffffffffd8,%r14
  40136b:	45 31 ff             	xor    %r15d,%r15d
  40136e:	c4 e2 7d 19 0d 99 3c 	vbroadcastsd 0x3c99(%rip),%ymm1        # 405010 <_IO_stdin_used+0x10>
  401375:	00 00 
  401377:	31 db                	xor    %ebx,%ebx
  401379:	eb 16                	jmp    401391 <main+0xe1>
  40137b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401380:	49 83 c6 28          	add    $0x28,%r14
  401384:	49 81 fe 18 78 7d 01 	cmp    $0x17d7818,%r14
  40138b:	0f 83 40 01 00 00    	jae    4014d1 <main+0x221>
  401391:	4b 8d 04 bf          	lea    (%r15,%r15,4),%rax
  401395:	48 c1 e0 06          	shl    $0x6,%rax
  401399:	48 8d 8c 24 18 12 00 	lea    0x1218(%rsp),%rcx
  4013a0:	00 
  4013a1:	c4 e2 7d 19 05 5e 3c 	vbroadcastsd 0x3c5e(%rip),%ymm0        # 405008 <_IO_stdin_used+0x8>
  4013a8:	00 00 
  4013aa:	c5 fd 11 04 01       	vmovupd %ymm0,(%rcx,%rax,1)
  4013af:	4c 8d 24 9b          	lea    (%rbx,%rbx,4),%r12
  4013b3:	49 c1 e4 06          	shl    $0x6,%r12
  4013b7:	c4 81 7d 11 4c 25 00 	vmovupd %ymm1,0x0(%r13,%r12,1)
  4013be:	c5 fd 11 44 01 20    	vmovupd %ymm0,0x20(%rcx,%rax,1)
  4013c4:	c4 81 7d 11 4c 25 20 	vmovupd %ymm1,0x20(%r13,%r12,1)
  4013cb:	c5 fd 11 44 01 40    	vmovupd %ymm0,0x40(%rcx,%rax,1)
  4013d1:	c4 81 7d 11 4c 25 40 	vmovupd %ymm1,0x40(%r13,%r12,1)
  4013d8:	c5 fd 11 44 01 60    	vmovupd %ymm0,0x60(%rcx,%rax,1)
  4013de:	c4 81 7d 11 4c 25 60 	vmovupd %ymm1,0x60(%r13,%r12,1)
  4013e5:	c5 fd 11 84 01 80 00 	vmovupd %ymm0,0x80(%rcx,%rax,1)
  4013ec:	00 00 
  4013ee:	c4 81 7d 11 8c 25 80 	vmovupd %ymm1,0x80(%r13,%r12,1)
  4013f5:	00 00 00 
  4013f8:	c5 fd 11 84 01 a0 00 	vmovupd %ymm0,0xa0(%rcx,%rax,1)
  4013ff:	00 00 
  401401:	c4 81 7d 11 8c 25 a0 	vmovupd %ymm1,0xa0(%r13,%r12,1)
  401408:	00 00 00 
  40140b:	c5 fd 11 84 01 c0 00 	vmovupd %ymm0,0xc0(%rcx,%rax,1)
  401412:	00 00 
  401414:	c4 81 7d 11 8c 25 c0 	vmovupd %ymm1,0xc0(%r13,%r12,1)
  40141b:	00 00 00 
  40141e:	c5 fd 11 84 01 e0 00 	vmovupd %ymm0,0xe0(%rcx,%rax,1)
  401425:	00 00 
  401427:	c4 81 7d 11 8c 25 e0 	vmovupd %ymm1,0xe0(%r13,%r12,1)
  40142e:	00 00 00 
  401431:	c5 fd 11 84 01 00 01 	vmovupd %ymm0,0x100(%rcx,%rax,1)
  401438:	00 00 
  40143a:	c4 81 7d 11 8c 25 00 	vmovupd %ymm1,0x100(%r13,%r12,1)
  401441:	01 00 00 
  401444:	c5 fd 11 84 01 20 01 	vmovupd %ymm0,0x120(%rcx,%rax,1)
  40144b:	00 00 
  40144d:	49 ff c7             	inc    %r15
  401450:	49 83 ff 0d          	cmp    $0xd,%r15
  401454:	74 1a                	je     401470 <main+0x1c0>
  401456:	c4 81 7d 11 8c 25 20 	vmovupd %ymm1,0x120(%r13,%r12,1)
  40145d:	01 00 00 
  401460:	48 ff c3             	inc    %rbx
  401463:	48 83 fb 0d          	cmp    $0xd,%rbx
  401467:	0f 85 13 ff ff ff    	jne    401380 <main+0xd0>
  40146d:	eb 3b                	jmp    4014aa <main+0x1fa>
  40146f:	90                   	nop
  401470:	be 40 10 00 00       	mov    $0x1040,%esi
  401475:	48 8d bc 24 c0 11 00 	lea    0x11c0(%rsp),%rdi
  40147c:	00 
  40147d:	31 d2                	xor    %edx,%edx
  40147f:	c5 f8 77             	vzeroupper
  401482:	e8 b9 0a 00 00       	call   401f40 <__libirc_nontemporal_store>
  401487:	c4 e2 7d 19 0d 80 3b 	vbroadcastsd 0x3b80(%rip),%ymm1        # 405010 <_IO_stdin_used+0x10>
  40148e:	00 00 
  401490:	45 31 ff             	xor    %r15d,%r15d
  401493:	c4 81 7d 11 8c 25 20 	vmovupd %ymm1,0x120(%r13,%r12,1)
  40149a:	01 00 00 
  40149d:	48 ff c3             	inc    %rbx
  4014a0:	48 83 fb 0d          	cmp    $0xd,%rbx
  4014a4:	0f 85 d6 fe ff ff    	jne    401380 <main+0xd0>
  4014aa:	be 40 10 00 00       	mov    $0x1040,%esi
  4014af:	48 8d bc 24 00 01 00 	lea    0x100(%rsp),%rdi
  4014b6:	00 
  4014b7:	31 d2                	xor    %edx,%edx
  4014b9:	c5 f8 77             	vzeroupper
  4014bc:	e8 7f 0a 00 00       	call   401f40 <__libirc_nontemporal_store>
  4014c1:	c4 e2 7d 19 0d 46 3b 	vbroadcastsd 0x3b46(%rip),%ymm1        # 405010 <_IO_stdin_used+0x10>
  4014c8:	00 00 
  4014ca:	31 db                	xor    %ebx,%ebx
  4014cc:	e9 af fe ff ff       	jmp    401380 <main+0xd0>
  4014d1:	48 c1 e3 06          	shl    $0x6,%rbx
  4014d5:	48 8d 34 9b          	lea    (%rbx,%rbx,4),%rsi
  4014d9:	48 8d bc 24 00 01 00 	lea    0x100(%rsp),%rdi
  4014e0:	00 
  4014e1:	ba 01 00 00 00       	mov    $0x1,%edx
  4014e6:	c5 f8 77             	vzeroupper
  4014e9:	e8 52 0a 00 00       	call   401f40 <__libirc_nontemporal_store>
  4014ee:	49 c1 e7 06          	shl    $0x6,%r15
  4014f2:	4b 8d 34 bf          	lea    (%r15,%r15,4),%rsi
  4014f6:	48 8d bc 24 c0 11 00 	lea    0x11c0(%rsp),%rdi
  4014fd:	00 
  4014fe:	ba 01 00 00 00       	mov    $0x1,%edx
  401503:	e8 38 0a 00 00       	call   401f40 <__libirc_nontemporal_store>
  401508:	0f ae f8             	sfence
  40150b:	e8 50 fb ff ff       	call   401060 <clock@plt>
  401510:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  401515:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40151a:	48 05 c0 d4 01 00    	add    $0x1d4c0,%rax
  401520:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  401525:	48 81 44 24 10 c0 d4 	addq   $0x1d4c0,0x10(%rsp)
  40152c:	01 00 
  40152e:	31 f6                	xor    %esi,%esi
  401530:	eb 2a                	jmp    40155c <main+0x2ac>
  401532:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  401539:	1f 84 00 00 00 00 00 
  401540:	48 81 44 24 18 00 10 	addq   $0x271000,0x18(%rsp)
  401547:	27 00 
  401549:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  40154e:	48 83 fe 4e          	cmp    $0x4e,%rsi
  401552:	48 8d 76 01          	lea    0x1(%rsi),%rsi
  401556:	0f 84 eb 02 00 00    	je     401847 <main+0x597>
  40155c:	48 89 74 24 40       	mov    %rsi,0x40(%rsp)
  401561:	48 c1 e6 06          	shl    $0x6,%rsi
  401565:	41 b8 87 13 00 00    	mov    $0x1387,%r8d
  40156b:	49 29 f0             	sub    %rsi,%r8
  40156e:	49 83 f8 40          	cmp    $0x40,%r8
  401572:	b8 3f 00 00 00       	mov    $0x3f,%eax
  401577:	4c 0f 4d c0          	cmovge %rax,%r8
  40157b:	49 ff c0             	inc    %r8
  40157e:	4c 89 44 24 58       	mov    %r8,0x58(%rsp)
  401583:	49 c1 e8 02          	shr    $0x2,%r8
  401587:	49 ff c8             	dec    %r8
  40158a:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  40158f:	45 31 db             	xor    %r11d,%r11d
  401592:	eb 27                	jmp    4015bb <main+0x30b>
  401594:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
  40159b:	00 00 00 00 00 
  4015a0:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
  4015a5:	48 81 c7 00 10 27 00 	add    $0x271000,%rdi
  4015ac:	4c 8b 5c 24 48       	mov    0x48(%rsp),%r11
  4015b1:	49 83 fb 4e          	cmp    $0x4e,%r11
  4015b5:	4d 8d 5b 01          	lea    0x1(%r11),%r11
  4015b9:	74 85                	je     401540 <main+0x290>
  4015bb:	4c 89 5c 24 48       	mov    %r11,0x48(%rsp)
  4015c0:	49 c1 e3 06          	shl    $0x6,%r11
  4015c4:	41 bf 87 13 00 00    	mov    $0x1387,%r15d
  4015ca:	4d 29 df             	sub    %r11,%r15
  4015cd:	49 83 ff 40          	cmp    $0x40,%r15
  4015d1:	b8 3f 00 00 00       	mov    $0x3f,%eax
  4015d6:	4c 0f 4d f8          	cmovge %rax,%r15
  4015da:	49 ff c7             	inc    %r15
  4015dd:	4d 89 fe             	mov    %r15,%r14
  4015e0:	49 c1 ee 02          	shr    $0x2,%r14
  4015e4:	49 ff ce             	dec    %r14
  4015e7:	48 89 7c 24 50       	mov    %rdi,0x50(%rsp)
  4015ec:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4015f1:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  4015f6:	31 c9                	xor    %ecx,%ecx
  4015f8:	eb 20                	jmp    40161a <main+0x36a>
  4015fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401600:	48 81 44 24 20 00 02 	addq   $0x200,0x20(%rsp)
  401607:	00 00 
  401609:	48 81 c7 00 02 00 00 	add    $0x200,%rdi
  401610:	48 83 f9 4e          	cmp    $0x4e,%rcx
  401614:	48 8d 49 01          	lea    0x1(%rcx),%rcx
  401618:	74 86                	je     4015a0 <main+0x2f0>
  40161a:	48 89 c8             	mov    %rcx,%rax
  40161d:	48 c1 e0 06          	shl    $0x6,%rax
  401621:	41 bd 87 13 00 00    	mov    $0x1387,%r13d
  401627:	49 29 c5             	sub    %rax,%r13
  40162a:	49 83 fd 40          	cmp    $0x40,%r13
  40162e:	b8 3f 00 00 00       	mov    $0x3f,%eax
  401633:	4c 0f 4d e8          	cmovge %rax,%r13
  401637:	48 83 7c 24 58 00    	cmpq   $0x0,0x58(%rsp)
  40163d:	74 c1                	je     401600 <main+0x350>
  40163f:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  401644:	45 31 d2             	xor    %r10d,%r10d
  401647:	eb 16                	jmp    40165f <main+0x3af>
  401649:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401650:	48 05 00 71 02 00    	add    $0x27100,%rax
  401656:	4d 39 c2             	cmp    %r8,%r10
  401659:	4d 8d 52 01          	lea    0x1(%r10),%r10
  40165d:	74 a1                	je     401600 <main+0x350>
  40165f:	4d 85 ff             	test   %r15,%r15
  401662:	74 ec                	je     401650 <main+0x3a0>
  401664:	4a 8d 14 96          	lea    (%rsi,%r10,4),%rdx
  401668:	48 69 d2 40 9c 00 00 	imul   $0x9c40,%rdx,%rdx
  40166f:	48 03 54 24 28       	add    0x28(%rsp),%rdx
  401674:	49 89 f9             	mov    %rdi,%r9
  401677:	31 db                	xor    %ebx,%ebx
  401679:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401680:	4d 8d 24 9b          	lea    (%r11,%rbx,4),%r12
  401684:	c4 a2 7d 19 04 e2    	vbroadcastsd (%rdx,%r12,8),%ymm0
  40168a:	c5 fc 11 84 24 e0 00 	vmovups %ymm0,0xe0(%rsp)
  401691:	00 00 
  401693:	c4 a2 7d 19 84 e2 40 	vbroadcastsd 0x9c40(%rdx,%r12,8),%ymm0
  40169a:	9c 00 00 
  40169d:	c5 fc 11 84 24 c0 00 	vmovups %ymm0,0xc0(%rsp)
  4016a4:	00 00 
  4016a6:	c4 a2 7d 19 84 e2 80 	vbroadcastsd 0x13880(%rdx,%r12,8),%ymm0
  4016ad:	38 01 00 
  4016b0:	c5 fc 11 84 24 a0 00 	vmovups %ymm0,0xa0(%rsp)
  4016b7:	00 00 
  4016b9:	c4 a2 7d 19 84 e2 c0 	vbroadcastsd 0x1d4c0(%rdx,%r12,8),%ymm0
  4016c0:	d4 01 00 
  4016c3:	c5 fc 11 84 24 80 00 	vmovups %ymm0,0x80(%rsp)
  4016ca:	00 00 
  4016cc:	c4 a2 7d 19 44 e2 08 	vbroadcastsd 0x8(%rdx,%r12,8),%ymm0
  4016d3:	c5 fc 11 44 24 60    	vmovups %ymm0,0x60(%rsp)
  4016d9:	c4 a2 7d 19 ac e2 48 	vbroadcastsd 0x9c48(%rdx,%r12,8),%ymm5
  4016e0:	9c 00 00 
  4016e3:	c4 a2 7d 19 b4 e2 88 	vbroadcastsd 0x13888(%rdx,%r12,8),%ymm6
  4016ea:	38 01 00 
  4016ed:	c4 a2 7d 19 bc e2 c8 	vbroadcastsd 0x1d4c8(%rdx,%r12,8),%ymm7
  4016f4:	d4 01 00 
  4016f7:	c4 22 7d 19 44 e2 10 	vbroadcastsd 0x10(%rdx,%r12,8),%ymm8
  4016fe:	c4 22 7d 19 8c e2 50 	vbroadcastsd 0x9c50(%rdx,%r12,8),%ymm9
  401705:	9c 00 00 
  401708:	c4 22 7d 19 94 e2 90 	vbroadcastsd 0x13890(%rdx,%r12,8),%ymm10
  40170f:	38 01 00 
  401712:	c4 22 7d 19 9c e2 d0 	vbroadcastsd 0x1d4d0(%rdx,%r12,8),%ymm11
  401719:	d4 01 00 
  40171c:	c4 22 7d 19 64 e2 18 	vbroadcastsd 0x18(%rdx,%r12,8),%ymm12
  401723:	c4 22 7d 19 ac e2 58 	vbroadcastsd 0x9c58(%rdx,%r12,8),%ymm13
  40172a:	9c 00 00 
  40172d:	c4 22 7d 19 b4 e2 98 	vbroadcastsd 0x13898(%rdx,%r12,8),%ymm14
  401734:	38 01 00 
  401737:	c4 22 7d 19 bc e2 d8 	vbroadcastsd 0x1d4d8(%rdx,%r12,8),%ymm15
  40173e:	d4 01 00 
  401741:	45 31 e4             	xor    %r12d,%r12d
  401744:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
  40174b:	00 00 00 00 00 
  401750:	c4 81 7d 10 84 e1 40 	vmovupd -0x1d4c0(%r9,%r12,8),%ymm0
  401757:	2b fe ff 
  40175a:	c4 a1 7d 10 8c e0 40 	vmovupd -0x1d4c0(%rax,%r12,8),%ymm1
  401761:	2b fe ff 
  401764:	c4 e2 fd b8 8c 24 e0 	vfmadd231pd 0xe0(%rsp),%ymm0,%ymm1
  40176b:	00 00 00 
  40176e:	c4 a1 7d 10 94 e0 80 	vmovupd -0x13880(%rax,%r12,8),%ymm2
  401775:	c7 fe ff 
  401778:	c4 e2 fd b8 94 24 c0 	vfmadd231pd 0xc0(%rsp),%ymm0,%ymm2
  40177f:	00 00 00 
  401782:	c4 a1 7d 10 9c e0 c0 	vmovupd -0x9c40(%rax,%r12,8),%ymm3
  401789:	63 ff ff 
  40178c:	c4 e2 fd b8 9c 24 a0 	vfmadd231pd 0xa0(%rsp),%ymm0,%ymm3
  401793:	00 00 00 
  401796:	c5 fd 10 a4 24 80 00 	vmovupd 0x80(%rsp),%ymm4
  40179d:	00 00 
  40179f:	c4 a2 dd a8 04 e0    	vfmadd213pd (%rax,%r12,8),%ymm4,%ymm0
  4017a5:	c4 81 7d 10 a4 e1 80 	vmovupd -0x13880(%r9,%r12,8),%ymm4
  4017ac:	c7 fe ff 
  4017af:	c4 e2 dd b8 4c 24 60 	vfmadd231pd 0x60(%rsp),%ymm4,%ymm1
  4017b6:	c4 e2 dd b8 d5       	vfmadd231pd %ymm5,%ymm4,%ymm2
  4017bb:	c4 e2 dd b8 de       	vfmadd231pd %ymm6,%ymm4,%ymm3
  4017c0:	c4 e2 c5 b8 c4       	vfmadd231pd %ymm4,%ymm7,%ymm0
  4017c5:	c4 81 7d 10 a4 e1 c0 	vmovupd -0x9c40(%r9,%r12,8),%ymm4
  4017cc:	63 ff ff 
  4017cf:	c4 c2 dd b8 c8       	vfmadd231pd %ymm8,%ymm4,%ymm1
  4017d4:	c4 c2 dd b8 d1       	vfmadd231pd %ymm9,%ymm4,%ymm2
  4017d9:	c4 c2 dd b8 da       	vfmadd231pd %ymm10,%ymm4,%ymm3
  4017de:	c4 e2 a5 b8 c4       	vfmadd231pd %ymm4,%ymm11,%ymm0
  4017e3:	c4 81 7d 10 24 e1    	vmovupd (%r9,%r12,8),%ymm4
  4017e9:	c4 c2 dd b8 cc       	vfmadd231pd %ymm12,%ymm4,%ymm1
  4017ee:	c4 a1 7d 11 8c e0 40 	vmovupd %ymm1,-0x1d4c0(%rax,%r12,8)
  4017f5:	2b fe ff 
  4017f8:	c4 c2 dd b8 d5       	vfmadd231pd %ymm13,%ymm4,%ymm2
  4017fd:	c4 a1 7d 11 94 e0 80 	vmovupd %ymm2,-0x13880(%rax,%r12,8)
  401804:	c7 fe ff 
  401807:	c4 c2 dd b8 de       	vfmadd231pd %ymm14,%ymm4,%ymm3
  40180c:	c4 a1 7d 11 9c e0 c0 	vmovupd %ymm3,-0x9c40(%rax,%r12,8)
  401813:	63 ff ff 
  401816:	c4 e2 85 b8 c4       	vfmadd231pd %ymm4,%ymm15,%ymm0
  40181b:	c4 a1 7d 11 04 e0    	vmovupd %ymm0,(%rax,%r12,8)
  401821:	49 83 c4 04          	add    $0x4,%r12
  401825:	4d 39 ec             	cmp    %r13,%r12
  401828:	0f 8e 22 ff ff ff    	jle    401750 <main+0x4a0>
  40182e:	49 81 c1 00 71 02 00 	add    $0x27100,%r9
  401835:	4c 39 f3             	cmp    %r14,%rbx
  401838:	48 8d 5b 01          	lea    0x1(%rbx),%rbx
  40183c:	0f 85 3e fe ff ff    	jne    401680 <main+0x3d0>
  401842:	e9 09 fe ff ff       	jmp    401650 <main+0x3a0>
  401847:	c5 f8 77             	vzeroupper
  40184a:	e8 11 f8 ff ff       	call   401060 <clock@plt>
  40184f:	48 2b 44 24 38       	sub    0x38(%rsp),%rax
  401854:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
  401858:	c4 e1 d3 2a c0       	vcvtsi2sd %rax,%xmm5,%xmm0
  40185d:	c5 fb 59 05 b3 37 00 	vmulsd 0x37b3(%rip),%xmm0,%xmm0        # 405018 <_IO_stdin_used+0x18>
  401864:	00 
  401865:	48 8b 3d 74 68 00 00 	mov    0x6874(%rip),%rdi        # 4080e0 <stderr@GLIBC_2.2.5>
  40186c:	be 20 50 40 00       	mov    $0x405020,%esi
  401871:	ba 88 13 00 00       	mov    $0x1388,%edx
  401876:	b0 01                	mov    $0x1,%al
  401878:	e8 63 f8 ff ff       	call   4010e0 <fprintf@plt>
  40187d:	bf 3b 50 40 00       	mov    $0x40503b,%edi
  401882:	be e6 51 40 00       	mov    $0x4051e6,%esi
  401887:	e8 c4 f8 ff ff       	call   401150 <fopen@plt>
  40188c:	48 85 c0             	test   %rax,%rax
  40188f:	0f 84 57 01 00 00    	je     4019ec <main+0x73c>
  401895:	49 89 c4             	mov    %rax,%r12
  401898:	45 31 f6             	xor    %r14d,%r14d
  40189b:	be 4d 50 40 00       	mov    $0x40504d,%esi
  4018a0:	48 89 c7             	mov    %rax,%rdi
  4018a3:	ba 88 13 00 00       	mov    $0x1388,%edx
  4018a8:	31 c0                	xor    %eax,%eax
  4018aa:	e8 31 f8 ff ff       	call   4010e0 <fprintf@plt>
  4018af:	4c 8b 7c 24 08       	mov    0x8(%rsp),%r15
  4018b4:	49 83 c7 38          	add    $0x38,%r15
  4018b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4018bf:	00 
  4018c0:	31 db                	xor    %ebx,%ebx
  4018c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  4018c9:	1f 84 00 00 00 00 00 
  4018d0:	c4 c1 7b 10 44 1f c8 	vmovsd -0x38(%r15,%rbx,1),%xmm0
  4018d7:	be 52 50 40 00       	mov    $0x405052,%esi
  4018dc:	4c 89 e7             	mov    %r12,%rdi
  4018df:	b0 01                	mov    $0x1,%al
  4018e1:	e8 fa f7 ff ff       	call   4010e0 <fprintf@plt>
  4018e6:	c4 c1 7b 10 44 1f d0 	vmovsd -0x30(%r15,%rbx,1),%xmm0
  4018ed:	be 52 50 40 00       	mov    $0x405052,%esi
  4018f2:	4c 89 e7             	mov    %r12,%rdi
  4018f5:	b0 01                	mov    $0x1,%al
  4018f7:	e8 e4 f7 ff ff       	call   4010e0 <fprintf@plt>
  4018fc:	c4 c1 7b 10 44 1f d8 	vmovsd -0x28(%r15,%rbx,1),%xmm0
  401903:	be 52 50 40 00       	mov    $0x405052,%esi
  401908:	4c 89 e7             	mov    %r12,%rdi
  40190b:	b0 01                	mov    $0x1,%al
  40190d:	e8 ce f7 ff ff       	call   4010e0 <fprintf@plt>
  401912:	c4 c1 7b 10 44 1f e0 	vmovsd -0x20(%r15,%rbx,1),%xmm0
  401919:	be 52 50 40 00       	mov    $0x405052,%esi
  40191e:	4c 89 e7             	mov    %r12,%rdi
  401921:	b0 01                	mov    $0x1,%al
  401923:	e8 b8 f7 ff ff       	call   4010e0 <fprintf@plt>
  401928:	c4 c1 7b 10 44 1f e8 	vmovsd -0x18(%r15,%rbx,1),%xmm0
  40192f:	be 52 50 40 00       	mov    $0x405052,%esi
  401934:	4c 89 e7             	mov    %r12,%rdi
  401937:	b0 01                	mov    $0x1,%al
  401939:	e8 a2 f7 ff ff       	call   4010e0 <fprintf@plt>
  40193e:	c4 c1 7b 10 44 1f f0 	vmovsd -0x10(%r15,%rbx,1),%xmm0
  401945:	be 52 50 40 00       	mov    $0x405052,%esi
  40194a:	4c 89 e7             	mov    %r12,%rdi
  40194d:	b0 01                	mov    $0x1,%al
  40194f:	e8 8c f7 ff ff       	call   4010e0 <fprintf@plt>
  401954:	c4 c1 7b 10 44 1f f8 	vmovsd -0x8(%r15,%rbx,1),%xmm0
  40195b:	be 52 50 40 00       	mov    $0x405052,%esi
  401960:	4c 89 e7             	mov    %r12,%rdi
  401963:	b0 01                	mov    $0x1,%al
  401965:	e8 76 f7 ff ff       	call   4010e0 <fprintf@plt>
  40196a:	c4 c1 7b 10 04 1f    	vmovsd (%r15,%rbx,1),%xmm0
  401970:	be 52 50 40 00       	mov    $0x405052,%esi
  401975:	4c 89 e7             	mov    %r12,%rdi
  401978:	b0 01                	mov    $0x1,%al
  40197a:	e8 61 f7 ff ff       	call   4010e0 <fprintf@plt>
  40197f:	48 83 c3 40          	add    $0x40,%rbx
  401983:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  40198a:	0f 85 40 ff ff ff    	jne    4018d0 <main+0x620>
  401990:	bf 0a 00 00 00       	mov    $0xa,%edi
  401995:	4c 89 e6             	mov    %r12,%rsi
  401998:	e8 23 f7 ff ff       	call   4010c0 <fputc@plt>
  40199d:	49 81 c7 40 9c 00 00 	add    $0x9c40,%r15
  4019a4:	49 81 fe e7 03 00 00 	cmp    $0x3e7,%r14
  4019ab:	4d 8d 76 01          	lea    0x1(%r14),%r14
  4019af:	0f 85 0b ff ff ff    	jne    4018c0 <main+0x610>
  4019b5:	4c 89 e7             	mov    %r12,%rdi
  4019b8:	e8 b3 f6 ff ff       	call   401070 <fclose@plt>
  4019bd:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
  4019c2:	e8 79 f6 ff ff       	call   401040 <free@plt>
  4019c7:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  4019cc:	e8 6f f6 ff ff       	call   401040 <free@plt>
  4019d1:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  4019d6:	e8 65 f6 ff ff       	call   401040 <free@plt>
  4019db:	31 c0                	xor    %eax,%eax
  4019dd:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  4019e1:	5b                   	pop    %rbx
  4019e2:	41 5c                	pop    %r12
  4019e4:	41 5d                	pop    %r13
  4019e6:	41 5e                	pop    %r14
  4019e8:	41 5f                	pop    %r15
  4019ea:	5d                   	pop    %rbp
  4019eb:	c3                   	ret
  4019ec:	bf 47 50 40 00       	mov    $0x405047,%edi
  4019f1:	e8 6a f7 ff ff       	call   401160 <perror@plt>
  4019f6:	b8 01 00 00 00       	mov    $0x1,%eax
  4019fb:	eb e0                	jmp    4019dd <main+0x72d>
  4019fd:	0f 1f 00             	nopl   (%rax)

0000000000401a00 <__libirc_nontemporal_store_512>:
  401a00:	f3 0f 1e fa          	endbr64
  401a04:	55                   	push   %rbp
  401a05:	41 57                	push   %r15
  401a07:	41 56                	push   %r14
  401a09:	41 54                	push   %r12
  401a0b:	53                   	push   %rbx
  401a0c:	49 89 f7             	mov    %rsi,%r15
  401a0f:	48 89 fb             	mov    %rdi,%rbx
  401a12:	4c 8d 77 58          	lea    0x58(%rdi),%r14
  401a16:	48 81 fe ff 00 00 00 	cmp    $0xff,%rsi
  401a1d:	77 2a                	ja     401a49 <__libirc_nontemporal_store_512+0x49>
  401a1f:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401a23:	48 8b 53 50          	mov    0x50(%rbx),%rdx
  401a27:	48 85 d2             	test   %rdx,%rdx
  401a2a:	0f 84 aa 00 00 00    	je     401ada <__libirc_nontemporal_store_512+0xda>
  401a30:	48 89 de             	mov    %rbx,%rsi
  401a33:	e8 b8 f6 ff ff       	call   4010f0 <memcpy@plt>
  401a38:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401a3c:	48 03 7b 50          	add    0x50(%rbx),%rdi
  401a40:	48 89 7b 40          	mov    %rdi,0x40(%rbx)
  401a44:	e9 91 00 00 00       	jmp    401ada <__libirc_nontemporal_store_512+0xda>
  401a49:	89 d5                	mov    %edx,%ebp
  401a4b:	48 8b 43 48          	mov    0x48(%rbx),%rax
  401a4f:	48 85 c0             	test   %rax,%rax
  401a52:	74 36                	je     401a8a <__libirc_nontemporal_store_512+0x8a>
  401a54:	83 e0 3f             	and    $0x3f,%eax
  401a57:	48 89 43 48          	mov    %rax,0x48(%rbx)
  401a5b:	41 bc 40 00 00 00    	mov    $0x40,%r12d
  401a61:	49 29 c4             	sub    %rax,%r12
  401a64:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401a68:	4c 89 f6             	mov    %r14,%rsi
  401a6b:	4c 89 e2             	mov    %r12,%rdx
  401a6e:	e8 7d f6 ff ff       	call   4010f0 <memcpy@plt>
  401a73:	48 c7 43 48 00 00 00 	movq   $0x0,0x48(%rbx)
  401a7a:	00 
  401a7b:	4c 01 63 40          	add    %r12,0x40(%rbx)
  401a7f:	4e 8d 34 23          	lea    (%rbx,%r12,1),%r14
  401a83:	49 83 c6 58          	add    $0x58,%r14
  401a87:	4d 29 e7             	sub    %r12,%r15
  401a8a:	48 8b 7b 50          	mov    0x50(%rbx),%rdi
  401a8e:	48 85 ff             	test   %rdi,%rdi
  401a91:	74 62                	je     401af5 <__libirc_nontemporal_store_512+0xf5>
  401a93:	41 bc 40 00 00 00    	mov    $0x40,%r12d
  401a99:	49 29 fc             	sub    %rdi,%r12
  401a9c:	48 01 df             	add    %rbx,%rdi
  401a9f:	4c 89 f6             	mov    %r14,%rsi
  401aa2:	4c 89 e2             	mov    %r12,%rdx
  401aa5:	e8 46 f6 ff ff       	call   4010f0 <memcpy@plt>
  401aaa:	48 8b 43 40          	mov    0x40(%rbx),%rax
  401aae:	62 f1 7c 48 10 03    	vmovups (%rbx),%zmm0
  401ab4:	62 f1 7c 48 2b 00    	vmovntps %zmm0,(%rax)
  401aba:	4d 01 e6             	add    %r12,%r14
  401abd:	4d 29 e7             	sub    %r12,%r15
  401ac0:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401ac4:	48 83 c7 40          	add    $0x40,%rdi
  401ac8:	48 89 7b 40          	mov    %rdi,0x40(%rbx)
  401acc:	49 83 ff 40          	cmp    $0x40,%r15
  401ad0:	73 27                	jae    401af9 <__libirc_nontemporal_store_512+0xf9>
  401ad2:	85 ed                	test   %ebp,%ebp
  401ad4:	0f 84 a0 01 00 00    	je     401c7a <__libirc_nontemporal_store_512+0x27a>
  401ada:	4c 89 f6             	mov    %r14,%rsi
  401add:	4c 89 fa             	mov    %r15,%rdx
  401ae0:	c5 f8 77             	vzeroupper
  401ae3:	e8 08 f6 ff ff       	call   4010f0 <memcpy@plt>
  401ae8:	4c 01 7b 40          	add    %r15,0x40(%rbx)
  401aec:	5b                   	pop    %rbx
  401aed:	41 5c                	pop    %r12
  401aef:	41 5e                	pop    %r14
  401af1:	41 5f                	pop    %r15
  401af3:	5d                   	pop    %rbp
  401af4:	c3                   	ret
  401af5:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401af9:	49 8d 47 c0          	lea    -0x40(%r15),%rax
  401afd:	48 83 f8 3f          	cmp    $0x3f,%rax
  401b01:	b9 3f 00 00 00       	mov    $0x3f,%ecx
  401b06:	48 0f 42 c8          	cmovb  %rax,%rcx
  401b0a:	48 f7 d1             	not    %rcx
  401b0d:	4c 01 f9             	add    %r15,%rcx
  401b10:	48 89 ca             	mov    %rcx,%rdx
  401b13:	48 c1 ea 06          	shr    $0x6,%rdx
  401b17:	48 8d 72 01          	lea    0x1(%rdx),%rsi
  401b1b:	48 81 f9 c0 01 00 00 	cmp    $0x1c0,%rcx
  401b22:	0f 82 f6 00 00 00    	jb     401c1e <__libirc_nontemporal_store_512+0x21e>
  401b28:	49 89 f0             	mov    %rsi,%r8
  401b2b:	49 c1 e8 03          	shr    $0x3,%r8
  401b2f:	4d 8d 8e c0 01 00 00 	lea    0x1c0(%r14),%r9
  401b36:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401b3d:	00 00 00 
  401b40:	62 d1 7c 48 10 41 f9 	vmovups -0x1c0(%r9),%zmm0
  401b47:	62 f1 7c 48 2b 07    	vmovntps %zmm0,(%rdi)
  401b4d:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401b51:	4c 8d 57 40          	lea    0x40(%rdi),%r10
  401b55:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401b59:	62 d1 7c 48 10 41 fa 	vmovups -0x180(%r9),%zmm0
  401b60:	62 f1 7c 48 2b 47 01 	vmovntps %zmm0,0x40(%rdi)
  401b67:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401b6b:	4c 8d 57 40          	lea    0x40(%rdi),%r10
  401b6f:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401b73:	62 d1 7c 48 10 41 fb 	vmovups -0x140(%r9),%zmm0
  401b7a:	62 f1 7c 48 2b 47 01 	vmovntps %zmm0,0x40(%rdi)
  401b81:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401b85:	4c 8d 57 40          	lea    0x40(%rdi),%r10
  401b89:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401b8d:	62 d1 7c 48 10 41 fc 	vmovups -0x100(%r9),%zmm0
  401b94:	62 f1 7c 48 2b 47 01 	vmovntps %zmm0,0x40(%rdi)
  401b9b:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401b9f:	4c 8d 57 40          	lea    0x40(%rdi),%r10
  401ba3:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401ba7:	62 d1 7c 48 10 41 fd 	vmovups -0xc0(%r9),%zmm0
  401bae:	62 f1 7c 48 2b 47 01 	vmovntps %zmm0,0x40(%rdi)
  401bb5:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401bb9:	4c 8d 57 40          	lea    0x40(%rdi),%r10
  401bbd:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401bc1:	62 d1 7c 48 10 41 fe 	vmovups -0x80(%r9),%zmm0
  401bc8:	62 f1 7c 48 2b 47 01 	vmovntps %zmm0,0x40(%rdi)
  401bcf:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401bd3:	4c 8d 57 40          	lea    0x40(%rdi),%r10
  401bd7:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401bdb:	62 d1 7c 48 10 41 ff 	vmovups -0x40(%r9),%zmm0
  401be2:	62 f1 7c 48 2b 47 01 	vmovntps %zmm0,0x40(%rdi)
  401be9:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401bed:	4c 8d 57 40          	lea    0x40(%rdi),%r10
  401bf1:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401bf5:	62 d1 7c 48 10 01    	vmovups (%r9),%zmm0
  401bfb:	62 f1 7c 48 2b 47 01 	vmovntps %zmm0,0x40(%rdi)
  401c02:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401c06:	48 83 c7 40          	add    $0x40,%rdi
  401c0a:	48 89 7b 40          	mov    %rdi,0x40(%rbx)
  401c0e:	49 81 c1 00 02 00 00 	add    $0x200,%r9
  401c15:	49 ff c8             	dec    %r8
  401c18:	0f 85 22 ff ff ff    	jne    401b40 <__libirc_nontemporal_store_512+0x140>
  401c1e:	48 83 e6 f8          	and    $0xfffffffffffffff8,%rsi
  401c22:	48 39 d6             	cmp    %rdx,%rsi
  401c25:	77 3a                	ja     401c61 <__libirc_nontemporal_store_512+0x261>
  401c27:	49 89 f0             	mov    %rsi,%r8
  401c2a:	49 c1 e0 06          	shl    $0x6,%r8
  401c2e:	4d 01 f0             	add    %r14,%r8
  401c31:	48 f7 d2             	not    %rdx
  401c34:	48 01 f2             	add    %rsi,%rdx
  401c37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  401c3e:	00 00 
  401c40:	62 d1 7c 48 10 00    	vmovups (%r8),%zmm0
  401c46:	62 f1 7c 48 2b 07    	vmovntps %zmm0,(%rdi)
  401c4c:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401c50:	48 83 c7 40          	add    $0x40,%rdi
  401c54:	48 89 7b 40          	mov    %rdi,0x40(%rbx)
  401c58:	49 83 c0 40          	add    $0x40,%r8
  401c5c:	48 ff c2             	inc    %rdx
  401c5f:	75 df                	jne    401c40 <__libirc_nontemporal_store_512+0x240>
  401c61:	48 83 e1 c0          	and    $0xffffffffffffffc0,%rcx
  401c65:	49 01 ce             	add    %rcx,%r14
  401c68:	49 83 c6 40          	add    $0x40,%r14
  401c6c:	48 29 c8             	sub    %rcx,%rax
  401c6f:	49 89 c7             	mov    %rax,%r15
  401c72:	85 ed                	test   %ebp,%ebp
  401c74:	0f 85 60 fe ff ff    	jne    401ada <__libirc_nontemporal_store_512+0xda>
  401c7a:	48 89 df             	mov    %rbx,%rdi
  401c7d:	4c 89 f6             	mov    %r14,%rsi
  401c80:	4c 89 fa             	mov    %r15,%rdx
  401c83:	c5 f8 77             	vzeroupper
  401c86:	e8 65 f4 ff ff       	call   4010f0 <memcpy@plt>
  401c8b:	4c 89 7b 50          	mov    %r15,0x50(%rbx)
  401c8f:	e9 58 fe ff ff       	jmp    401aec <__libirc_nontemporal_store_512+0xec>
  401c94:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401c9b:	00 00 00 
  401c9e:	66 90                	xchg   %ax,%ax

0000000000401ca0 <__libirc_nontemporal_store_256>:
  401ca0:	f3 0f 1e fa          	endbr64
  401ca4:	55                   	push   %rbp
  401ca5:	41 57                	push   %r15
  401ca7:	41 56                	push   %r14
  401ca9:	41 54                	push   %r12
  401cab:	53                   	push   %rbx
  401cac:	49 89 f7             	mov    %rsi,%r15
  401caf:	48 89 fb             	mov    %rdi,%rbx
  401cb2:	4c 8d 77 58          	lea    0x58(%rdi),%r14
  401cb6:	48 83 fe 7f          	cmp    $0x7f,%rsi
  401cba:	77 2a                	ja     401ce6 <__libirc_nontemporal_store_256+0x46>
  401cbc:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401cc0:	48 8b 53 50          	mov    0x50(%rbx),%rdx
  401cc4:	48 85 d2             	test   %rdx,%rdx
  401cc7:	0f 84 a6 00 00 00    	je     401d73 <__libirc_nontemporal_store_256+0xd3>
  401ccd:	48 89 de             	mov    %rbx,%rsi
  401cd0:	e8 1b f4 ff ff       	call   4010f0 <memcpy@plt>
  401cd5:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401cd9:	48 03 7b 50          	add    0x50(%rbx),%rdi
  401cdd:	48 89 7b 40          	mov    %rdi,0x40(%rbx)
  401ce1:	e9 8d 00 00 00       	jmp    401d73 <__libirc_nontemporal_store_256+0xd3>
  401ce6:	89 d5                	mov    %edx,%ebp
  401ce8:	48 8b 43 48          	mov    0x48(%rbx),%rax
  401cec:	48 85 c0             	test   %rax,%rax
  401cef:	74 36                	je     401d27 <__libirc_nontemporal_store_256+0x87>
  401cf1:	83 e0 1f             	and    $0x1f,%eax
  401cf4:	48 89 43 48          	mov    %rax,0x48(%rbx)
  401cf8:	41 bc 20 00 00 00    	mov    $0x20,%r12d
  401cfe:	49 29 c4             	sub    %rax,%r12
  401d01:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401d05:	4c 89 f6             	mov    %r14,%rsi
  401d08:	4c 89 e2             	mov    %r12,%rdx
  401d0b:	e8 e0 f3 ff ff       	call   4010f0 <memcpy@plt>
  401d10:	48 c7 43 48 00 00 00 	movq   $0x0,0x48(%rbx)
  401d17:	00 
  401d18:	4c 01 63 40          	add    %r12,0x40(%rbx)
  401d1c:	4e 8d 34 23          	lea    (%rbx,%r12,1),%r14
  401d20:	49 83 c6 58          	add    $0x58,%r14
  401d24:	4d 29 e7             	sub    %r12,%r15
  401d27:	48 8b 7b 50          	mov    0x50(%rbx),%rdi
  401d2b:	48 85 ff             	test   %rdi,%rdi
  401d2e:	74 5e                	je     401d8e <__libirc_nontemporal_store_256+0xee>
  401d30:	41 bc 20 00 00 00    	mov    $0x20,%r12d
  401d36:	49 29 fc             	sub    %rdi,%r12
  401d39:	48 01 df             	add    %rbx,%rdi
  401d3c:	4c 89 f6             	mov    %r14,%rsi
  401d3f:	4c 89 e2             	mov    %r12,%rdx
  401d42:	e8 a9 f3 ff ff       	call   4010f0 <memcpy@plt>
  401d47:	48 8b 43 40          	mov    0x40(%rbx),%rax
  401d4b:	c5 fc 10 03          	vmovups (%rbx),%ymm0
  401d4f:	c5 fc 2b 00          	vmovntps %ymm0,(%rax)
  401d53:	4d 01 e6             	add    %r12,%r14
  401d56:	4d 29 e7             	sub    %r12,%r15
  401d59:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401d5d:	48 83 c7 20          	add    $0x20,%rdi
  401d61:	48 89 7b 40          	mov    %rdi,0x40(%rbx)
  401d65:	49 83 ff 20          	cmp    $0x20,%r15
  401d69:	73 27                	jae    401d92 <__libirc_nontemporal_store_256+0xf2>
  401d6b:	85 ed                	test   %ebp,%ebp
  401d6d:	0f 84 84 01 00 00    	je     401ef7 <__libirc_nontemporal_store_256+0x257>
  401d73:	4c 89 f6             	mov    %r14,%rsi
  401d76:	4c 89 fa             	mov    %r15,%rdx
  401d79:	c5 f8 77             	vzeroupper
  401d7c:	e8 6f f3 ff ff       	call   4010f0 <memcpy@plt>
  401d81:	4c 01 7b 40          	add    %r15,0x40(%rbx)
  401d85:	5b                   	pop    %rbx
  401d86:	41 5c                	pop    %r12
  401d88:	41 5e                	pop    %r14
  401d8a:	41 5f                	pop    %r15
  401d8c:	5d                   	pop    %rbp
  401d8d:	c3                   	ret
  401d8e:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401d92:	49 8d 47 e0          	lea    -0x20(%r15),%rax
  401d96:	48 83 f8 1f          	cmp    $0x1f,%rax
  401d9a:	b9 1f 00 00 00       	mov    $0x1f,%ecx
  401d9f:	48 0f 42 c8          	cmovb  %rax,%rcx
  401da3:	48 f7 d1             	not    %rcx
  401da6:	4c 01 f9             	add    %r15,%rcx
  401da9:	48 89 ca             	mov    %rcx,%rdx
  401dac:	48 c1 ea 05          	shr    $0x5,%rdx
  401db0:	48 8d 72 01          	lea    0x1(%rdx),%rsi
  401db4:	48 81 f9 e0 00 00 00 	cmp    $0xe0,%rcx
  401dbb:	0f 82 de 00 00 00    	jb     401e9f <__libirc_nontemporal_store_256+0x1ff>
  401dc1:	49 89 f0             	mov    %rsi,%r8
  401dc4:	49 c1 e8 03          	shr    $0x3,%r8
  401dc8:	4d 8d 8e e0 00 00 00 	lea    0xe0(%r14),%r9
  401dcf:	90                   	nop
  401dd0:	c4 c1 7c 10 81 20 ff 	vmovups -0xe0(%r9),%ymm0
  401dd7:	ff ff 
  401dd9:	c5 fc 2b 07          	vmovntps %ymm0,(%rdi)
  401ddd:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401de1:	4c 8d 57 20          	lea    0x20(%rdi),%r10
  401de5:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401de9:	c4 c1 7c 10 81 40 ff 	vmovups -0xc0(%r9),%ymm0
  401df0:	ff ff 
  401df2:	c5 fc 2b 47 20       	vmovntps %ymm0,0x20(%rdi)
  401df7:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401dfb:	4c 8d 57 20          	lea    0x20(%rdi),%r10
  401dff:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401e03:	c4 c1 7c 10 81 60 ff 	vmovups -0xa0(%r9),%ymm0
  401e0a:	ff ff 
  401e0c:	c5 fc 2b 47 20       	vmovntps %ymm0,0x20(%rdi)
  401e11:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401e15:	4c 8d 57 20          	lea    0x20(%rdi),%r10
  401e19:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401e1d:	c4 c1 7c 10 41 80    	vmovups -0x80(%r9),%ymm0
  401e23:	c5 fc 2b 47 20       	vmovntps %ymm0,0x20(%rdi)
  401e28:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401e2c:	4c 8d 57 20          	lea    0x20(%rdi),%r10
  401e30:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401e34:	c4 c1 7c 10 41 a0    	vmovups -0x60(%r9),%ymm0
  401e3a:	c5 fc 2b 47 20       	vmovntps %ymm0,0x20(%rdi)
  401e3f:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401e43:	4c 8d 57 20          	lea    0x20(%rdi),%r10
  401e47:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401e4b:	c4 c1 7c 10 41 c0    	vmovups -0x40(%r9),%ymm0
  401e51:	c5 fc 2b 47 20       	vmovntps %ymm0,0x20(%rdi)
  401e56:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401e5a:	4c 8d 57 20          	lea    0x20(%rdi),%r10
  401e5e:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401e62:	c4 c1 7c 10 41 e0    	vmovups -0x20(%r9),%ymm0
  401e68:	c5 fc 2b 47 20       	vmovntps %ymm0,0x20(%rdi)
  401e6d:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401e71:	4c 8d 57 20          	lea    0x20(%rdi),%r10
  401e75:	4c 89 53 40          	mov    %r10,0x40(%rbx)
  401e79:	c4 c1 7c 10 01       	vmovups (%r9),%ymm0
  401e7e:	c5 fc 2b 47 20       	vmovntps %ymm0,0x20(%rdi)
  401e83:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401e87:	48 83 c7 20          	add    $0x20,%rdi
  401e8b:	48 89 7b 40          	mov    %rdi,0x40(%rbx)
  401e8f:	49 81 c1 00 01 00 00 	add    $0x100,%r9
  401e96:	49 ff c8             	dec    %r8
  401e99:	0f 85 31 ff ff ff    	jne    401dd0 <__libirc_nontemporal_store_256+0x130>
  401e9f:	48 83 e6 f8          	and    $0xfffffffffffffff8,%rsi
  401ea3:	48 39 d6             	cmp    %rdx,%rsi
  401ea6:	77 36                	ja     401ede <__libirc_nontemporal_store_256+0x23e>
  401ea8:	49 89 f0             	mov    %rsi,%r8
  401eab:	49 c1 e0 05          	shl    $0x5,%r8
  401eaf:	4d 01 f0             	add    %r14,%r8
  401eb2:	48 f7 d2             	not    %rdx
  401eb5:	48 01 f2             	add    %rsi,%rdx
  401eb8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401ebf:	00 
  401ec0:	c4 c1 7c 10 00       	vmovups (%r8),%ymm0
  401ec5:	c5 fc 2b 07          	vmovntps %ymm0,(%rdi)
  401ec9:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  401ecd:	48 83 c7 20          	add    $0x20,%rdi
  401ed1:	48 89 7b 40          	mov    %rdi,0x40(%rbx)
  401ed5:	49 83 c0 20          	add    $0x20,%r8
  401ed9:	48 ff c2             	inc    %rdx
  401edc:	75 e2                	jne    401ec0 <__libirc_nontemporal_store_256+0x220>
  401ede:	48 83 e1 e0          	and    $0xffffffffffffffe0,%rcx
  401ee2:	49 01 ce             	add    %rcx,%r14
  401ee5:	49 83 c6 20          	add    $0x20,%r14
  401ee9:	48 29 c8             	sub    %rcx,%rax
  401eec:	49 89 c7             	mov    %rax,%r15
  401eef:	85 ed                	test   %ebp,%ebp
  401ef1:	0f 85 7c fe ff ff    	jne    401d73 <__libirc_nontemporal_store_256+0xd3>
  401ef7:	48 89 df             	mov    %rbx,%rdi
  401efa:	4c 89 f6             	mov    %r14,%rsi
  401efd:	4c 89 fa             	mov    %r15,%rdx
  401f00:	c5 f8 77             	vzeroupper
  401f03:	e8 e8 f1 ff ff       	call   4010f0 <memcpy@plt>
  401f08:	4c 89 7b 50          	mov    %r15,0x50(%rbx)
  401f0c:	e9 74 fe ff ff       	jmp    401d85 <__libirc_nontemporal_store_256+0xe5>
  401f11:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401f18:	00 00 00 
  401f1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401f20 <__libirc_get_nontemporal_store_func.V>:
  401f20:	48 8d 05 79 fd ff ff 	lea    -0x287(%rip),%rax        # 401ca0 <__libirc_nontemporal_store_256>
  401f27:	c3                   	ret
  401f28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401f2f:	00 

0000000000401f30 <__libirc_get_nontemporal_store_func.a>:
  401f30:	48 8d 05 c9 fa ff ff 	lea    -0x537(%rip),%rax        # 401a00 <__libirc_nontemporal_store_512>
  401f37:	c3                   	ret
  401f38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401f3f:	00 

0000000000401f40 <__libirc_nontemporal_store>:
  401f40:	f3 0f 1e fa          	endbr64
  401f44:	41 57                	push   %r15
  401f46:	41 56                	push   %r14
  401f48:	53                   	push   %rbx
  401f49:	89 d3                	mov    %edx,%ebx
  401f4b:	49 89 f6             	mov    %rsi,%r14
  401f4e:	49 89 ff             	mov    %rdi,%r15
  401f51:	48 8b 05 98 61 00 00 	mov    0x6198(%rip),%rax        # 4080f0 <__libirc_nontemporal_store._impl_func>
  401f58:	48 85 c0             	test   %rax,%rax
  401f5b:	75 5a                	jne    401fb7 <__libirc_nontemporal_store+0x77>
  401f5d:	48 c7 c1 10 81 40 00 	mov    $0x408110,%rcx
  401f64:	48 8b 01             	mov    (%rcx),%rax
  401f67:	48 85 c0             	test   %rax,%rax
  401f6a:	75 11                	jne    401f7d <__libirc_nontemporal_store+0x3d>
  401f6c:	0f 1f 40 00          	nopl   0x0(%rax)
  401f70:	e8 2b 10 00 00       	call   402fa0 <__intel_cpu_features_init_x>
  401f75:	48 8b 01             	mov    (%rcx),%rax
  401f78:	48 85 c0             	test   %rax,%rax
  401f7b:	74 f3                	je     401f70 <__libirc_nontemporal_store+0x30>
  401f7d:	48 89 c1             	mov    %rax,%rcx
  401f80:	48 f7 d1             	not    %rcx
  401f83:	48 ba ec 9f 9d 19 64 	movabs $0x64199d9fec,%rdx
  401f8a:	00 00 00 
  401f8d:	48 85 d1             	test   %rdx,%rcx
  401f90:	75 07                	jne    401f99 <__libirc_nontemporal_store+0x59>
  401f92:	e8 99 ff ff ff       	call   401f30 <__libirc_get_nontemporal_store_func.a>
  401f97:	eb 17                	jmp    401fb0 <__libirc_nontemporal_store+0x70>
  401f99:	f7 d0                	not    %eax
  401f9b:	a9 ec 9f 9d 00       	test   $0x9d9fec,%eax
  401fa0:	75 07                	jne    401fa9 <__libirc_nontemporal_store+0x69>
  401fa2:	e8 79 ff ff ff       	call   401f20 <__libirc_get_nontemporal_store_func.V>
  401fa7:	eb 07                	jmp    401fb0 <__libirc_nontemporal_store+0x70>
  401fa9:	48 8d 05 20 00 00 00 	lea    0x20(%rip),%rax        # 401fd0 <__libirc_nontemporal_store_fallback>
  401fb0:	48 89 05 39 61 00 00 	mov    %rax,0x6139(%rip)        # 4080f0 <__libirc_nontemporal_store._impl_func>
  401fb7:	4c 89 ff             	mov    %r15,%rdi
  401fba:	4c 89 f6             	mov    %r14,%rsi
  401fbd:	89 da                	mov    %ebx,%edx
  401fbf:	5b                   	pop    %rbx
  401fc0:	41 5e                	pop    %r14
  401fc2:	41 5f                	pop    %r15
  401fc4:	ff e0                	jmp    *%rax
  401fc6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401fcd:	00 00 00 

0000000000401fd0 <__libirc_nontemporal_store_fallback>:
  401fd0:	f3 0f 1e fa          	endbr64
  401fd4:	41 56                	push   %r14
  401fd6:	53                   	push   %rbx
  401fd7:	50                   	push   %rax
  401fd8:	48 89 f3             	mov    %rsi,%rbx
  401fdb:	49 89 fe             	mov    %rdi,%r14
  401fde:	48 8b 7f 40          	mov    0x40(%rdi),%rdi
  401fe2:	49 8d 76 58          	lea    0x58(%r14),%rsi
  401fe6:	48 89 da             	mov    %rbx,%rdx
  401fe9:	e8 02 f1 ff ff       	call   4010f0 <memcpy@plt>
  401fee:	49 01 5e 40          	add    %rbx,0x40(%r14)
  401ff2:	48 83 c4 08          	add    $0x8,%rsp
  401ff6:	5b                   	pop    %rbx
  401ff7:	41 5e                	pop    %r14
  401ff9:	c3                   	ret
  401ffa:	66 90                	xchg   %ax,%ax
  401ffc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000402000 <__intel_new_feature_proc_init_n>:
  402000:	f3 0f 1e fa          	endbr64
  402004:	55                   	push   %rbp
  402005:	41 57                	push   %r15
  402007:	41 56                	push   %r14
  402009:	41 55                	push   %r13
  40200b:	41 54                	push   %r12
  40200d:	53                   	push   %rbx
  40200e:	48 81 ec 38 04 00 00 	sub    $0x438,%rsp
  402015:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40201c:	00 00 
  40201e:	48 89 84 24 30 04 00 	mov    %rax,0x430(%rsp)
  402025:	00 
  402026:	0f 57 c0             	xorps  %xmm0,%xmm0
  402029:	0f 29 44 24 10       	movaps %xmm0,0x10(%rsp)
  40202e:	48 c7 c1 00 81 40 00 	mov    $0x408100,%rcx
  402035:	48 83 39 00          	cmpq   $0x0,(%rcx)
  402039:	75 17                	jne    402052 <__intel_new_feature_proc_init_n+0x52>
  40203b:	e8 80 04 00 00       	call   4024c0 <__intel_cpu_features_init>
  402040:	85 c0                	test   %eax,%eax
  402042:	0f 85 0b 02 00 00    	jne    402253 <__intel_new_feature_proc_init_n+0x253>
  402048:	48 83 39 00          	cmpq   $0x0,(%rcx)
  40204c:	0f 84 01 02 00 00    	je     402253 <__intel_new_feature_proc_init_n+0x253>
  402052:	83 ff 02             	cmp    $0x2,%edi
  402055:	7d 38                	jge    40208f <__intel_new_feature_proc_init_n+0x8f>
  402057:	48 63 c7             	movslq %edi,%rax
  40205a:	48 8b 0c c1          	mov    (%rcx,%rax,8),%rcx
  40205e:	48 f7 d1             	not    %rcx
  402061:	48 85 ce             	test   %rcx,%rsi
  402064:	75 48                	jne    4020ae <__intel_new_feature_proc_init_n+0xae>
  402066:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40206d:	00 00 
  40206f:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  402076:	00 
  402077:	0f 85 e8 02 00 00    	jne    402365 <__intel_new_feature_proc_init_n+0x365>
  40207d:	48 81 c4 38 04 00 00 	add    $0x438,%rsp
  402084:	5b                   	pop    %rbx
  402085:	41 5c                	pop    %r12
  402087:	41 5d                	pop    %r13
  402089:	41 5e                	pop    %r14
  40208b:	41 5f                	pop    %r15
  40208d:	5d                   	pop    %rbp
  40208e:	c3                   	ret
  40208f:	bf 01 00 00 00       	mov    $0x1,%edi
  402094:	31 f6                	xor    %esi,%esi
  402096:	31 d2                	xor    %edx,%edx
  402098:	31 c0                	xor    %eax,%eax
  40209a:	e8 b1 1e 00 00       	call   403f50 <__libirc_print>
  40209f:	bf 01 00 00 00       	mov    $0x1,%edi
  4020a4:	be 3a 00 00 00       	mov    $0x3a,%esi
  4020a9:	e9 bf 01 00 00       	jmp    40226d <__intel_new_feature_proc_init_n+0x26d>
  4020ae:	48 21 f1             	and    %rsi,%rcx
  4020b1:	48 89 4c c4 10       	mov    %rcx,0x10(%rsp,%rax,8)
  4020b6:	45 31 ff             	xor    %r15d,%r15d
  4020b9:	bf 39 00 00 00       	mov    $0x39,%edi
  4020be:	31 f6                	xor    %esi,%esi
  4020c0:	31 c0                	xor    %eax,%eax
  4020c2:	e8 49 1c 00 00       	call   403d10 <__libirc_get_msg>
  4020c7:	48 89 04 24          	mov    %rax,(%rsp)
  4020cb:	c6 44 24 30 00       	movb   $0x0,0x30(%rsp)
  4020d0:	bd 01 00 00 00       	mov    $0x1,%ebp
  4020d5:	4c 8d 64 24 10       	lea    0x10(%rsp),%r12
  4020da:	4c 8d 6c 24 30       	lea    0x30(%rsp),%r13
  4020df:	31 db                	xor    %ebx,%ebx
  4020e1:	eb 31                	jmp    402114 <__intel_new_feature_proc_init_n+0x114>
  4020e3:	b8 ff 03 00 00       	mov    $0x3ff,%eax
  4020e8:	44 29 f8             	sub    %r15d,%eax
  4020eb:	48 63 d0             	movslq %eax,%rdx
  4020ee:	b9 00 04 00 00       	mov    $0x400,%ecx
  4020f3:	4c 89 ef             	mov    %r13,%rdi
  4020f6:	4c 89 f6             	mov    %r14,%rsi
  4020f9:	e8 a2 f0 ff ff       	call   4011a0 <__strncat_chk@plt>
  4020fe:	4c 89 ef             	mov    %r13,%rdi
  402101:	e8 7a ef ff ff       	call   401080 <strlen@plt>
  402106:	49 89 c7             	mov    %rax,%r15
  402109:	ff c5                	inc    %ebp
  40210b:	83 fd 47             	cmp    $0x47,%ebp
  40210e:	0f 84 26 01 00 00    	je     40223a <__intel_new_feature_proc_init_n+0x23a>
  402114:	89 e8                	mov    %ebp,%eax
  402116:	e8 95 19 00 00       	call   403ab0 <__libirc_get_feature_bitpos>
  40211b:	85 c0                	test   %eax,%eax
  40211d:	78 ea                	js     402109 <__intel_new_feature_proc_init_n+0x109>
  40211f:	4c 89 e7             	mov    %r12,%rdi
  402122:	89 ee                	mov    %ebp,%esi
  402124:	e8 c7 19 00 00       	call   403af0 <__libirc_get_cpu_feature>
  402129:	85 c0                	test   %eax,%eax
  40212b:	74 dc                	je     402109 <__intel_new_feature_proc_init_n+0x109>
  40212d:	4c 89 e7             	mov    %r12,%rdi
  402130:	89 ee                	mov    %ebp,%esi
  402132:	e8 b9 19 00 00       	call   403af0 <__libirc_get_cpu_feature>
  402137:	85 c0                	test   %eax,%eax
  402139:	0f 88 e6 00 00 00    	js     402225 <__intel_new_feature_proc_init_n+0x225>
  40213f:	89 ef                	mov    %ebp,%edi
  402141:	e8 7a 0e 00 00       	call   402fc0 <__libirc_get_feature_name>
  402146:	48 85 c0             	test   %rax,%rax
  402149:	0f 84 d6 00 00 00    	je     402225 <__intel_new_feature_proc_init_n+0x225>
  40214f:	49 89 c6             	mov    %rax,%r14
  402152:	80 38 00             	cmpb   $0x0,(%rax)
  402155:	0f 84 ca 00 00 00    	je     402225 <__intel_new_feature_proc_init_n+0x225>
  40215b:	80 7c 24 30 00       	cmpb   $0x0,0x30(%rsp)
  402160:	74 81                	je     4020e3 <__intel_new_feature_proc_init_n+0xe3>
  402162:	48 85 db             	test   %rbx,%rbx
  402165:	0f 84 b2 00 00 00    	je     40221d <__intel_new_feature_proc_init_n+0x21d>
  40216b:	4d 89 ec             	mov    %r13,%r12
  40216e:	48 89 df             	mov    %rbx,%rdi
  402171:	e8 0a ef ff ff       	call   401080 <strlen@plt>
  402176:	49 89 c5             	mov    %rax,%r13
  402179:	48 8d 3d d8 2e 00 00 	lea    0x2ed8(%rip),%rdi        # 405058 <_IO_stdin_used+0x58>
  402180:	e8 fb ee ff ff       	call   401080 <strlen@plt>
  402185:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  40218a:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  40218f:	49 63 df             	movslq %r15d,%rbx
  402192:	48 8b 3c 24          	mov    (%rsp),%rdi
  402196:	e8 e5 ee ff ff       	call   401080 <strlen@plt>
  40219b:	49 89 c7             	mov    %rax,%r15
  40219e:	4c 89 f7             	mov    %r14,%rdi
  4021a1:	e8 da ee ff ff       	call   401080 <strlen@plt>
  4021a6:	49 01 dd             	add    %rbx,%r13
  4021a9:	4c 03 6c 24 28       	add    0x28(%rsp),%r13
  4021ae:	4c 01 f8             	add    %r15,%rax
  4021b1:	4c 01 e8             	add    %r13,%rax
  4021b4:	b9 ff 03 00 00       	mov    $0x3ff,%ecx
  4021b9:	29 d9                	sub    %ebx,%ecx
  4021bb:	48 63 d1             	movslq %ecx,%rdx
  4021be:	48 3d ff 03 00 00    	cmp    $0x3ff,%rax
  4021c4:	0f 87 dd 00 00 00    	ja     4022a7 <__intel_new_feature_proc_init_n+0x2a7>
  4021ca:	b9 00 04 00 00       	mov    $0x400,%ecx
  4021cf:	4d 89 e5             	mov    %r12,%r13
  4021d2:	4c 89 e7             	mov    %r12,%rdi
  4021d5:	48 8d 35 7c 2e 00 00 	lea    0x2e7c(%rip),%rsi        # 405058 <_IO_stdin_used+0x58>
  4021dc:	e8 bf ef ff ff       	call   4011a0 <__strncat_chk@plt>
  4021e1:	4c 89 e7             	mov    %r12,%rdi
  4021e4:	e8 97 ee ff ff       	call   401080 <strlen@plt>
  4021e9:	48 c1 e0 20          	shl    $0x20,%rax
  4021ed:	48 ba 00 00 00 00 ff 	movabs $0x3ff00000000,%rdx
  4021f4:	03 00 00 
  4021f7:	48 29 c2             	sub    %rax,%rdx
  4021fa:	48 c1 fa 20          	sar    $0x20,%rdx
  4021fe:	b9 00 04 00 00       	mov    $0x400,%ecx
  402203:	4c 89 e7             	mov    %r12,%rdi
  402206:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  40220b:	e8 90 ef ff ff       	call   4011a0 <__strncat_chk@plt>
  402210:	4c 89 f3             	mov    %r14,%rbx
  402213:	4c 8d 64 24 10       	lea    0x10(%rsp),%r12
  402218:	e9 e1 fe ff ff       	jmp    4020fe <__intel_new_feature_proc_init_n+0xfe>
  40221d:	4c 89 f3             	mov    %r14,%rbx
  402220:	e9 e4 fe ff ff       	jmp    402109 <__intel_new_feature_proc_init_n+0x109>
  402225:	bf 01 00 00 00       	mov    $0x1,%edi
  40222a:	31 f6                	xor    %esi,%esi
  40222c:	31 d2                	xor    %edx,%edx
  40222e:	31 c0                	xor    %eax,%eax
  402230:	e8 1b 1d 00 00       	call   403f50 <__libirc_print>
  402235:	e9 ce 00 00 00       	jmp    402308 <__intel_new_feature_proc_init_n+0x308>
  40223a:	48 85 db             	test   %rbx,%rbx
  40223d:	0f 84 ac 00 00 00    	je     4022ef <__intel_new_feature_proc_init_n+0x2ef>
  402243:	49 89 dc             	mov    %rbx,%r12
  402246:	b8 ff 03 00 00       	mov    $0x3ff,%eax
  40224b:	44 29 f8             	sub    %r15d,%eax
  40224e:	48 63 d0             	movslq %eax,%rdx
  402251:	eb 59                	jmp    4022ac <__intel_new_feature_proc_init_n+0x2ac>
  402253:	bf 01 00 00 00       	mov    $0x1,%edi
  402258:	31 f6                	xor    %esi,%esi
  40225a:	31 d2                	xor    %edx,%edx
  40225c:	31 c0                	xor    %eax,%eax
  40225e:	e8 ed 1c 00 00       	call   403f50 <__libirc_print>
  402263:	bf 01 00 00 00       	mov    $0x1,%edi
  402268:	be 3b 00 00 00       	mov    $0x3b,%esi
  40226d:	31 d2                	xor    %edx,%edx
  40226f:	31 c0                	xor    %eax,%eax
  402271:	e8 da 1c 00 00       	call   403f50 <__libirc_print>
  402276:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40227d:	00 00 
  40227f:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  402286:	00 
  402287:	0f 85 d8 00 00 00    	jne    402365 <__intel_new_feature_proc_init_n+0x365>
  40228d:	bf 01 00 00 00       	mov    $0x1,%edi
  402292:	31 f6                	xor    %esi,%esi
  402294:	31 d2                	xor    %edx,%edx
  402296:	31 c0                	xor    %eax,%eax
  402298:	e8 b3 1c 00 00       	call   403f50 <__libirc_print>
  40229d:	bf 01 00 00 00       	mov    $0x1,%edi
  4022a2:	e8 d9 ee ff ff       	call   401180 <exit@plt>
  4022a7:	4c 8b 64 24 08       	mov    0x8(%rsp),%r12
  4022ac:	4c 8d 74 24 30       	lea    0x30(%rsp),%r14
  4022b1:	b9 00 04 00 00       	mov    $0x400,%ecx
  4022b6:	4c 89 f7             	mov    %r14,%rdi
  4022b9:	48 8b 34 24          	mov    (%rsp),%rsi
  4022bd:	e8 de ee ff ff       	call   4011a0 <__strncat_chk@plt>
  4022c2:	4c 89 f7             	mov    %r14,%rdi
  4022c5:	e8 b6 ed ff ff       	call   401080 <strlen@plt>
  4022ca:	48 c1 e0 20          	shl    $0x20,%rax
  4022ce:	48 ba 00 00 00 00 ff 	movabs $0x3ff00000000,%rdx
  4022d5:	03 00 00 
  4022d8:	48 29 c2             	sub    %rax,%rdx
  4022db:	48 c1 fa 20          	sar    $0x20,%rdx
  4022df:	b9 00 04 00 00       	mov    $0x400,%ecx
  4022e4:	4c 89 f7             	mov    %r14,%rdi
  4022e7:	4c 89 e6             	mov    %r12,%rsi
  4022ea:	e8 b1 ee ff ff       	call   4011a0 <__strncat_chk@plt>
  4022ef:	0f b6 5c 24 30       	movzbl 0x30(%rsp),%ebx
  4022f4:	bf 01 00 00 00       	mov    $0x1,%edi
  4022f9:	31 f6                	xor    %esi,%esi
  4022fb:	31 d2                	xor    %edx,%edx
  4022fd:	31 c0                	xor    %eax,%eax
  4022ff:	e8 4c 1c 00 00       	call   403f50 <__libirc_print>
  402304:	84 db                	test   %bl,%bl
  402306:	75 15                	jne    40231d <__intel_new_feature_proc_init_n+0x31d>
  402308:	bf 01 00 00 00       	mov    $0x1,%edi
  40230d:	be 3a 00 00 00       	mov    $0x3a,%esi
  402312:	31 d2                	xor    %edx,%edx
  402314:	31 c0                	xor    %eax,%eax
  402316:	e8 35 1c 00 00       	call   403f50 <__libirc_print>
  40231b:	eb 1b                	jmp    402338 <__intel_new_feature_proc_init_n+0x338>
  40231d:	48 8d 4c 24 30       	lea    0x30(%rsp),%rcx
  402322:	bf 01 00 00 00       	mov    $0x1,%edi
  402327:	be 38 00 00 00       	mov    $0x38,%esi
  40232c:	ba 01 00 00 00       	mov    $0x1,%edx
  402331:	31 c0                	xor    %eax,%eax
  402333:	e8 18 1c 00 00       	call   403f50 <__libirc_print>
  402338:	bf 01 00 00 00       	mov    $0x1,%edi
  40233d:	31 f6                	xor    %esi,%esi
  40233f:	31 d2                	xor    %edx,%edx
  402341:	31 c0                	xor    %eax,%eax
  402343:	e8 08 1c 00 00       	call   403f50 <__libirc_print>
  402348:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40234f:	00 00 
  402351:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  402358:	00 
  402359:	75 0a                	jne    402365 <__intel_new_feature_proc_init_n+0x365>
  40235b:	bf 01 00 00 00       	mov    $0x1,%edi
  402360:	e8 1b ee ff ff       	call   401180 <exit@plt>
  402365:	e8 26 ed ff ff       	call   401090 <__stack_chk_fail@plt>
  40236a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000402370 <__intel_new_feature_proc_init>:
  402370:	f3 0f 1e fa          	endbr64
  402374:	53                   	push   %rbx
  402375:	89 fb                	mov    %edi,%ebx
  402377:	31 ff                	xor    %edi,%edi
  402379:	e8 82 fc ff ff       	call   402000 <__intel_new_feature_proc_init_n>
  40237e:	48 c7 c7 00 81 40 00 	mov    $0x408100,%rdi
  402385:	be 06 00 00 00       	mov    $0x6,%esi
  40238a:	e8 61 17 00 00       	call   403af0 <__libirc_get_cpu_feature>
  40238f:	83 f8 01             	cmp    $0x1,%eax
  402392:	75 0a                	jne    40239e <__intel_new_feature_proc_init+0x2e>
  402394:	31 ff                	xor    %edi,%edi
  402396:	89 de                	mov    %ebx,%esi
  402398:	5b                   	pop    %rbx
  402399:	e9 52 00 00 00       	jmp    4023f0 <__intel_proc_init_ftzdazule>
  40239e:	85 c0                	test   %eax,%eax
  4023a0:	78 02                	js     4023a4 <__intel_new_feature_proc_init+0x34>
  4023a2:	5b                   	pop    %rbx
  4023a3:	c3                   	ret
  4023a4:	bf 01 00 00 00       	mov    $0x1,%edi
  4023a9:	31 f6                	xor    %esi,%esi
  4023ab:	31 d2                	xor    %edx,%edx
  4023ad:	31 c0                	xor    %eax,%eax
  4023af:	e8 9c 1b 00 00       	call   403f50 <__libirc_print>
  4023b4:	bf 01 00 00 00       	mov    $0x1,%edi
  4023b9:	be 3a 00 00 00       	mov    $0x3a,%esi
  4023be:	31 d2                	xor    %edx,%edx
  4023c0:	31 c0                	xor    %eax,%eax
  4023c2:	e8 89 1b 00 00       	call   403f50 <__libirc_print>
  4023c7:	bf 01 00 00 00       	mov    $0x1,%edi
  4023cc:	31 f6                	xor    %esi,%esi
  4023ce:	31 d2                	xor    %edx,%edx
  4023d0:	31 c0                	xor    %eax,%eax
  4023d2:	e8 79 1b 00 00       	call   403f50 <__libirc_print>
  4023d7:	bf 01 00 00 00       	mov    $0x1,%edi
  4023dc:	e8 9f ed ff ff       	call   401180 <exit@plt>
  4023e1:	0f 1f 00             	nopl   (%rax)
  4023e4:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4023eb:	00 00 00 
  4023ee:	66 90                	xchg   %ax,%ax

00000000004023f0 <__intel_proc_init_ftzdazule>:
  4023f0:	f3 0f 1e fa          	endbr64
  4023f4:	55                   	push   %rbp
  4023f5:	41 56                	push   %r14
  4023f7:	53                   	push   %rbx
  4023f8:	48 81 ec 20 02 00 00 	sub    $0x220,%rsp
  4023ff:	89 f3                	mov    %esi,%ebx
  402401:	41 89 f6             	mov    %esi,%r14d
  402404:	41 83 e6 04          	and    $0x4,%r14d
  402408:	89 f5                	mov    %esi,%ebp
  40240a:	83 e5 02             	and    $0x2,%ebp
  40240d:	74 07                	je     402416 <__intel_proc_init_ftzdazule+0x26>
  40240f:	89 f8                	mov    %edi,%eax
  402411:	83 e0 02             	and    $0x2,%eax
  402414:	74 12                	je     402428 <__intel_proc_init_ftzdazule+0x38>
  402416:	31 c0                	xor    %eax,%eax
  402418:	45 85 f6             	test   %r14d,%r14d
  40241b:	74 38                	je     402455 <__intel_proc_init_ftzdazule+0x65>
  40241d:	b8 01 00 00 00       	mov    $0x1,%eax
  402422:	40 f6 c7 04          	test   $0x4,%dil
  402426:	75 2d                	jne    402455 <__intel_proc_init_ftzdazule+0x65>
  402428:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  40242d:	ba 00 02 00 00       	mov    $0x200,%edx
  402432:	31 f6                	xor    %esi,%esi
  402434:	e8 77 ec ff ff       	call   4010b0 <memset@plt>
  402439:	0f ae 44 24 20       	fxsave 0x20(%rsp)
  40243e:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
  402442:	89 c1                	mov    %eax,%ecx
  402444:	c1 e1 19             	shl    $0x19,%ecx
  402447:	c1 f9 1f             	sar    $0x1f,%ecx
  40244a:	21 cd                	and    %ecx,%ebp
  40244c:	c1 e0 0e             	shl    $0xe,%eax
  40244f:	c1 f8 1f             	sar    $0x1f,%eax
  402452:	44 21 f0             	and    %r14d,%eax
  402455:	f6 c3 01             	test   $0x1,%bl
  402458:	74 17                	je     402471 <__intel_proc_init_ftzdazule+0x81>
  40245a:	0f ae 5c 24 1c       	stmxcsr 0x1c(%rsp)
  40245f:	b9 00 80 00 00       	mov    $0x8000,%ecx
  402464:	0b 4c 24 1c          	or     0x1c(%rsp),%ecx
  402468:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
  40246c:	0f ae 54 24 18       	ldmxcsr 0x18(%rsp)
  402471:	85 ed                	test   %ebp,%ebp
  402473:	74 15                	je     40248a <__intel_proc_init_ftzdazule+0x9a>
  402475:	0f ae 5c 24 14       	stmxcsr 0x14(%rsp)
  40247a:	8b 4c 24 14          	mov    0x14(%rsp),%ecx
  40247e:	83 c9 40             	or     $0x40,%ecx
  402481:	89 4c 24 10          	mov    %ecx,0x10(%rsp)
  402485:	0f ae 54 24 10       	ldmxcsr 0x10(%rsp)
  40248a:	85 c0                	test   %eax,%eax
  40248c:	74 17                	je     4024a5 <__intel_proc_init_ftzdazule+0xb5>
  40248e:	0f ae 5c 24 0c       	stmxcsr 0xc(%rsp)
  402493:	b8 00 00 02 00       	mov    $0x20000,%eax
  402498:	0b 44 24 0c          	or     0xc(%rsp),%eax
  40249c:	89 44 24 08          	mov    %eax,0x8(%rsp)
  4024a0:	0f ae 54 24 08       	ldmxcsr 0x8(%rsp)
  4024a5:	48 81 c4 20 02 00 00 	add    $0x220,%rsp
  4024ac:	5b                   	pop    %rbx
  4024ad:	41 5e                	pop    %r14
  4024af:	5d                   	pop    %rbp
  4024b0:	c3                   	ret
  4024b1:	0f 1f 00             	nopl   (%rax)
  4024b4:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4024bb:	00 00 00 
  4024be:	66 90                	xchg   %ax,%ax

00000000004024c0 <__intel_cpu_features_init>:
  4024c0:	f3 0f 1e fa          	endbr64
  4024c4:	50                   	push   %rax
  4024c5:	b8 01 00 00 00       	mov    $0x1,%eax
  4024ca:	e8 11 00 00 00       	call   4024e0 <__intel_cpu_features_init_body>
  4024cf:	48 83 c4 08          	add    $0x8,%rsp
  4024d3:	c3                   	ret
  4024d4:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4024db:	00 00 00 
  4024de:	66 90                	xchg   %ax,%ax

00000000004024e0 <__intel_cpu_features_init_body>:
  4024e0:	41 53                	push   %r11
  4024e2:	41 52                	push   %r10
  4024e4:	41 51                	push   %r9
  4024e6:	41 50                	push   %r8
  4024e8:	52                   	push   %rdx
  4024e9:	51                   	push   %rcx
  4024ea:	56                   	push   %rsi
  4024eb:	57                   	push   %rdi
  4024ec:	55                   	push   %rbp
  4024ed:	53                   	push   %rbx
  4024ee:	48 81 ec 38 01 00 00 	sub    $0x138,%rsp
  4024f5:	44 0f 29 bc 24 20 01 	movaps %xmm15,0x120(%rsp)
  4024fc:	00 00 
  4024fe:	44 0f 29 b4 24 10 01 	movaps %xmm14,0x110(%rsp)
  402505:	00 00 
  402507:	44 0f 29 ac 24 00 01 	movaps %xmm13,0x100(%rsp)
  40250e:	00 00 
  402510:	44 0f 29 a4 24 f0 00 	movaps %xmm12,0xf0(%rsp)
  402517:	00 00 
  402519:	44 0f 29 9c 24 e0 00 	movaps %xmm11,0xe0(%rsp)
  402520:	00 00 
  402522:	44 0f 29 94 24 d0 00 	movaps %xmm10,0xd0(%rsp)
  402529:	00 00 
  40252b:	44 0f 29 8c 24 c0 00 	movaps %xmm9,0xc0(%rsp)
  402532:	00 00 
  402534:	44 0f 29 84 24 b0 00 	movaps %xmm8,0xb0(%rsp)
  40253b:	00 00 
  40253d:	0f 29 bc 24 a0 00 00 	movaps %xmm7,0xa0(%rsp)
  402544:	00 
  402545:	0f 29 b4 24 90 00 00 	movaps %xmm6,0x90(%rsp)
  40254c:	00 
  40254d:	0f 29 ac 24 80 00 00 	movaps %xmm5,0x80(%rsp)
  402554:	00 
  402555:	0f 29 64 24 70       	movaps %xmm4,0x70(%rsp)
  40255a:	0f 29 5c 24 60       	movaps %xmm3,0x60(%rsp)
  40255f:	0f 29 54 24 50       	movaps %xmm2,0x50(%rsp)
  402564:	0f 29 4c 24 40       	movaps %xmm1,0x40(%rsp)
  402569:	0f 29 44 24 30       	movaps %xmm0,0x30(%rsp)
  40256e:	89 c5                	mov    %eax,%ebp
  402570:	0f 57 c0             	xorps  %xmm0,%xmm0
  402573:	0f 29 04 24          	movaps %xmm0,(%rsp)
  402577:	0f 29 44 24 20       	movaps %xmm0,0x20(%rsp)
  40257c:	48 89 e0             	mov    %rsp,%rax
  40257f:	b9 01 00 00 00       	mov    $0x1,%ecx
  402584:	e8 b7 15 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402589:	85 c0                	test   %eax,%eax
  40258b:	0f 85 81 03 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402591:	31 c0                	xor    %eax,%eax
  402593:	0f a2                	cpuid
  402595:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  402599:	89 5c 24 18          	mov    %ebx,0x18(%rsp)
  40259d:	89 4c 24 14          	mov    %ecx,0x14(%rsp)
  4025a1:	89 54 24 10          	mov    %edx,0x10(%rsp)
  4025a5:	83 7c 24 1c 00       	cmpl   $0x0,0x1c(%rsp)
  4025aa:	0f 84 55 03 00 00    	je     402905 <__intel_cpu_features_init_body+0x425>
  4025b0:	83 fd 01             	cmp    $0x1,%ebp
  4025b3:	75 2a                	jne    4025df <__intel_cpu_features_init_body+0xff>
  4025b5:	81 7c 24 18 47 65 6e 	cmpl   $0x756e6547,0x18(%rsp)
  4025bc:	75 
  4025bd:	0f 85 42 03 00 00    	jne    402905 <__intel_cpu_features_init_body+0x425>
  4025c3:	81 7c 24 10 69 6e 65 	cmpl   $0x49656e69,0x10(%rsp)
  4025ca:	49 
  4025cb:	0f 85 34 03 00 00    	jne    402905 <__intel_cpu_features_init_body+0x425>
  4025d1:	81 7c 24 14 6e 74 65 	cmpl   $0x6c65746e,0x14(%rsp)
  4025d8:	6c 
  4025d9:	0f 85 26 03 00 00    	jne    402905 <__intel_cpu_features_init_body+0x425>
  4025df:	b8 01 00 00 00       	mov    $0x1,%eax
  4025e4:	0f a2                	cpuid
  4025e6:	41 89 d2             	mov    %edx,%r10d
  4025e9:	41 89 c8             	mov    %ecx,%r8d
  4025ec:	41 f6 c2 01          	test   $0x1,%r10b
  4025f0:	74 15                	je     402607 <__intel_cpu_features_init_body+0x127>
  4025f2:	48 89 e0             	mov    %rsp,%rax
  4025f5:	b9 02 00 00 00       	mov    $0x2,%ecx
  4025fa:	e8 41 15 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4025ff:	85 c0                	test   %eax,%eax
  402601:	0f 85 0b 03 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402607:	66 45 85 d2          	test   %r10w,%r10w
  40260b:	79 15                	jns    402622 <__intel_cpu_features_init_body+0x142>
  40260d:	48 89 e0             	mov    %rsp,%rax
  402610:	b9 03 00 00 00       	mov    $0x3,%ecx
  402615:	e8 26 15 00 00       	call   403b40 <__libirc_set_cpu_feature>
  40261a:	85 c0                	test   %eax,%eax
  40261c:	0f 85 f0 02 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402622:	41 f7 c2 00 00 80 00 	test   $0x800000,%r10d
  402629:	74 15                	je     402640 <__intel_cpu_features_init_body+0x160>
  40262b:	48 89 e0             	mov    %rsp,%rax
  40262e:	b9 04 00 00 00       	mov    $0x4,%ecx
  402633:	e8 08 15 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402638:	85 c0                	test   %eax,%eax
  40263a:	0f 85 d2 02 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402640:	41 f7 c2 00 00 00 01 	test   $0x1000000,%r10d
  402647:	0f 85 54 03 00 00    	jne    4029a1 <__intel_cpu_features_init_body+0x4c1>
  40264d:	41 f7 c0 00 00 00 40 	test   $0x40000000,%r8d
  402654:	74 15                	je     40266b <__intel_cpu_features_init_body+0x18b>
  402656:	48 89 e0             	mov    %rsp,%rax
  402659:	b9 12 00 00 00       	mov    $0x12,%ecx
  40265e:	e8 dd 14 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402663:	85 c0                	test   %eax,%eax
  402665:	0f 85 a7 02 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  40266b:	41 f7 c2 00 00 00 01 	test   $0x1000000,%r10d
  402672:	75 10                	jne    402684 <__intel_cpu_features_init_body+0x1a4>
  402674:	b8 07 00 00 00       	mov    $0x7,%eax
  402679:	31 c9                	xor    %ecx,%ecx
  40267b:	0f a2                	cpuid
  40267d:	89 cf                	mov    %ecx,%edi
  40267f:	89 d6                	mov    %edx,%esi
  402681:	41 89 d9             	mov    %ebx,%r9d
  402684:	44 89 c8             	mov    %r9d,%eax
  402687:	f7 d0                	not    %eax
  402689:	a9 08 01 00 00       	test   $0x108,%eax
  40268e:	75 15                	jne    4026a5 <__intel_cpu_features_init_body+0x1c5>
  402690:	48 89 e0             	mov    %rsp,%rax
  402693:	b9 14 00 00 00       	mov    $0x14,%ecx
  402698:	e8 a3 14 00 00       	call   403b40 <__libirc_set_cpu_feature>
  40269d:	85 c0                	test   %eax,%eax
  40269f:	0f 85 6d 02 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  4026a5:	41 f6 c1 04          	test   $0x4,%r9b
  4026a9:	74 15                	je     4026c0 <__intel_cpu_features_init_body+0x1e0>
  4026ab:	48 89 e0             	mov    %rsp,%rax
  4026ae:	b9 36 00 00 00       	mov    $0x36,%ecx
  4026b3:	e8 88 14 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4026b8:	85 c0                	test   %eax,%eax
  4026ba:	0f 85 52 02 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  4026c0:	41 f6 c1 10          	test   $0x10,%r9b
  4026c4:	74 15                	je     4026db <__intel_cpu_features_init_body+0x1fb>
  4026c6:	48 89 e0             	mov    %rsp,%rax
  4026c9:	b9 16 00 00 00       	mov    $0x16,%ecx
  4026ce:	e8 6d 14 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4026d3:	85 c0                	test   %eax,%eax
  4026d5:	0f 85 37 02 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  4026db:	41 f7 c1 00 08 00 00 	test   $0x800,%r9d
  4026e2:	74 15                	je     4026f9 <__intel_cpu_features_init_body+0x219>
  4026e4:	48 89 e0             	mov    %rsp,%rax
  4026e7:	b9 17 00 00 00       	mov    $0x17,%ecx
  4026ec:	e8 4f 14 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4026f1:	85 c0                	test   %eax,%eax
  4026f3:	0f 85 19 02 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  4026f9:	41 f7 c1 00 00 08 00 	test   $0x80000,%r9d
  402700:	74 15                	je     402717 <__intel_cpu_features_init_body+0x237>
  402702:	48 89 e0             	mov    %rsp,%rax
  402705:	b9 1d 00 00 00       	mov    $0x1d,%ecx
  40270a:	e8 31 14 00 00       	call   403b40 <__libirc_set_cpu_feature>
  40270f:	85 c0                	test   %eax,%eax
  402711:	0f 85 fb 01 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402717:	41 f7 c1 00 00 04 00 	test   $0x40000,%r9d
  40271e:	74 15                	je     402735 <__intel_cpu_features_init_body+0x255>
  402720:	48 89 e0             	mov    %rsp,%rax
  402723:	b9 1e 00 00 00       	mov    $0x1e,%ecx
  402728:	e8 13 14 00 00       	call   403b40 <__libirc_set_cpu_feature>
  40272d:	85 c0                	test   %eax,%eax
  40272f:	0f 85 dd 01 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402735:	41 f7 c1 00 00 00 01 	test   $0x1000000,%r9d
  40273c:	74 15                	je     402753 <__intel_cpu_features_init_body+0x273>
  40273e:	48 89 e0             	mov    %rsp,%rax
  402741:	b9 32 00 00 00       	mov    $0x32,%ecx
  402746:	e8 f5 13 00 00       	call   403b40 <__libirc_set_cpu_feature>
  40274b:	85 c0                	test   %eax,%eax
  40274d:	0f 85 bf 01 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402753:	b8 01 00 00 80       	mov    $0x80000001,%eax
  402758:	0f a2                	cpuid
  40275a:	f6 c1 20             	test   $0x20,%cl
  40275d:	74 15                	je     402774 <__intel_cpu_features_init_body+0x294>
  40275f:	48 89 e0             	mov    %rsp,%rax
  402762:	b9 15 00 00 00       	mov    $0x15,%ecx
  402767:	e8 d4 13 00 00       	call   403b40 <__libirc_set_cpu_feature>
  40276c:	85 c0                	test   %eax,%eax
  40276e:	0f 85 9e 01 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402774:	b8 08 00 00 80       	mov    $0x80000008,%eax
  402779:	0f a2                	cpuid
  40277b:	f7 c3 00 02 00 00    	test   $0x200,%ebx
  402781:	74 15                	je     402798 <__intel_cpu_features_init_body+0x2b8>
  402783:	48 89 e0             	mov    %rsp,%rax
  402786:	b9 37 00 00 00       	mov    $0x37,%ecx
  40278b:	e8 b0 13 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402790:	85 c0                	test   %eax,%eax
  402792:	0f 85 7a 01 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402798:	40 f6 c7 20          	test   $0x20,%dil
  40279c:	74 15                	je     4027b3 <__intel_cpu_features_init_body+0x2d3>
  40279e:	48 89 e0             	mov    %rsp,%rax
  4027a1:	b9 3e 00 00 00       	mov    $0x3e,%ecx
  4027a6:	e8 95 13 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4027ab:	85 c0                	test   %eax,%eax
  4027ad:	0f 85 5f 01 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  4027b3:	40 84 ff             	test   %dil,%dil
  4027b6:	79 15                	jns    4027cd <__intel_cpu_features_init_body+0x2ed>
  4027b8:	48 89 e0             	mov    %rsp,%rax
  4027bb:	b9 35 00 00 00       	mov    $0x35,%ecx
  4027c0:	e8 7b 13 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4027c5:	85 c0                	test   %eax,%eax
  4027c7:	0f 85 45 01 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  4027cd:	f7 c7 00 01 00 00    	test   $0x100,%edi
  4027d3:	74 15                	je     4027ea <__intel_cpu_features_init_body+0x30a>
  4027d5:	48 89 e0             	mov    %rsp,%rax
  4027d8:	b9 2e 00 00 00       	mov    $0x2e,%ecx
  4027dd:	e8 5e 13 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4027e2:	85 c0                	test   %eax,%eax
  4027e4:	0f 85 28 01 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  4027ea:	f7 c7 00 00 40 00    	test   $0x400000,%edi
  4027f0:	74 15                	je     402807 <__intel_cpu_features_init_body+0x327>
  4027f2:	48 89 e0             	mov    %rsp,%rax
  4027f5:	b9 33 00 00 00       	mov    $0x33,%ecx
  4027fa:	e8 41 13 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4027ff:	85 c0                	test   %eax,%eax
  402801:	0f 85 0b 01 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402807:	f7 c7 00 00 00 01    	test   $0x1000000,%edi
  40280d:	74 15                	je     402824 <__intel_cpu_features_init_body+0x344>
  40280f:	48 89 e0             	mov    %rsp,%rax
  402812:	b9 3b 00 00 00       	mov    $0x3b,%ecx
  402817:	e8 24 13 00 00       	call   403b40 <__libirc_set_cpu_feature>
  40281c:	85 c0                	test   %eax,%eax
  40281e:	0f 85 ee 00 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402824:	f7 c7 00 00 00 08    	test   $0x8000000,%edi
  40282a:	74 15                	je     402841 <__intel_cpu_features_init_body+0x361>
  40282c:	48 89 e0             	mov    %rsp,%rax
  40282f:	b9 3c 00 00 00       	mov    $0x3c,%ecx
  402834:	e8 07 13 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402839:	85 c0                	test   %eax,%eax
  40283b:	0f 85 d1 00 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402841:	f7 c7 00 00 00 10    	test   $0x10000000,%edi
  402847:	74 15                	je     40285e <__intel_cpu_features_init_body+0x37e>
  402849:	48 89 e0             	mov    %rsp,%rax
  40284c:	b9 3d 00 00 00       	mov    $0x3d,%ecx
  402851:	e8 ea 12 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402856:	85 c0                	test   %eax,%eax
  402858:	0f 85 b4 00 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  40285e:	f7 c7 00 00 00 20    	test   $0x20000000,%edi
  402864:	74 15                	je     40287b <__intel_cpu_features_init_body+0x39b>
  402866:	48 89 e0             	mov    %rsp,%rax
  402869:	b9 40 00 00 00       	mov    $0x40,%ecx
  40286e:	e8 cd 12 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402873:	85 c0                	test   %eax,%eax
  402875:	0f 85 97 00 00 00    	jne    402912 <__intel_cpu_features_init_body+0x432>
  40287b:	f7 c6 00 00 10 00    	test   $0x100000,%esi
  402881:	74 11                	je     402894 <__intel_cpu_features_init_body+0x3b4>
  402883:	48 89 e0             	mov    %rsp,%rax
  402886:	b9 34 00 00 00       	mov    $0x34,%ecx
  40288b:	e8 b0 12 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402890:	85 c0                	test   %eax,%eax
  402892:	75 7e                	jne    402912 <__intel_cpu_features_init_body+0x432>
  402894:	f7 c6 00 00 04 00    	test   $0x40000,%esi
  40289a:	74 11                	je     4028ad <__intel_cpu_features_init_body+0x3cd>
  40289c:	48 89 e0             	mov    %rsp,%rax
  40289f:	b9 38 00 00 00       	mov    $0x38,%ecx
  4028a4:	e8 97 12 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4028a9:	85 c0                	test   %eax,%eax
  4028ab:	75 65                	jne    402912 <__intel_cpu_features_init_body+0x432>
  4028ad:	b8 14 00 00 00       	mov    $0x14,%eax
  4028b2:	31 c9                	xor    %ecx,%ecx
  4028b4:	0f a2                	cpuid
  4028b6:	f6 c3 10             	test   $0x10,%bl
  4028b9:	74 11                	je     4028cc <__intel_cpu_features_init_body+0x3ec>
  4028bb:	48 89 e0             	mov    %rsp,%rax
  4028be:	b9 1b 00 00 00       	mov    $0x1b,%ecx
  4028c3:	e8 78 12 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4028c8:	85 c0                	test   %eax,%eax
  4028ca:	75 46                	jne    402912 <__intel_cpu_features_init_body+0x432>
  4028cc:	f7 c7 00 00 80 00    	test   $0x800000,%edi
  4028d2:	0f 85 3a 02 00 00    	jne    402b12 <__intel_cpu_features_init_body+0x632>
  4028d8:	41 f7 c0 00 00 00 08 	test   $0x8000000,%r8d
  4028df:	0f 85 88 02 00 00    	jne    402b6d <__intel_cpu_features_init_body+0x68d>
  4028e5:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  4028ea:	e8 b1 12 00 00       	call   403ba0 <__libirc_handle_intel_isa_disable>
  4028ef:	85 c0                	test   %eax,%eax
  4028f1:	0f 8e 09 06 00 00    	jle    402f00 <__intel_cpu_features_init_body+0xa20>
  4028f7:	0f 28 44 24 20       	movaps 0x20(%rsp),%xmm0
  4028fc:	0f 55 04 24          	andnps (%rsp),%xmm0
  402900:	e9 ff 05 00 00       	jmp    402f04 <__intel_cpu_features_init_body+0xa24>
  402905:	0f 28 04 24          	movaps (%rsp),%xmm0
  402909:	0f 29 05 f0 57 00 00 	movaps %xmm0,0x57f0(%rip)        # 408100 <__intel_cpu_feature_indicator>
  402910:	31 c0                	xor    %eax,%eax
  402912:	0f 28 44 24 30       	movaps 0x30(%rsp),%xmm0
  402917:	0f 28 4c 24 40       	movaps 0x40(%rsp),%xmm1
  40291c:	0f 28 54 24 50       	movaps 0x50(%rsp),%xmm2
  402921:	0f 28 5c 24 60       	movaps 0x60(%rsp),%xmm3
  402926:	0f 28 64 24 70       	movaps 0x70(%rsp),%xmm4
  40292b:	0f 28 ac 24 80 00 00 	movaps 0x80(%rsp),%xmm5
  402932:	00 
  402933:	0f 28 b4 24 90 00 00 	movaps 0x90(%rsp),%xmm6
  40293a:	00 
  40293b:	0f 28 bc 24 a0 00 00 	movaps 0xa0(%rsp),%xmm7
  402942:	00 
  402943:	44 0f 28 84 24 b0 00 	movaps 0xb0(%rsp),%xmm8
  40294a:	00 00 
  40294c:	44 0f 28 8c 24 c0 00 	movaps 0xc0(%rsp),%xmm9
  402953:	00 00 
  402955:	44 0f 28 94 24 d0 00 	movaps 0xd0(%rsp),%xmm10
  40295c:	00 00 
  40295e:	44 0f 28 9c 24 e0 00 	movaps 0xe0(%rsp),%xmm11
  402965:	00 00 
  402967:	44 0f 28 a4 24 f0 00 	movaps 0xf0(%rsp),%xmm12
  40296e:	00 00 
  402970:	44 0f 28 ac 24 00 01 	movaps 0x100(%rsp),%xmm13
  402977:	00 00 
  402979:	44 0f 28 b4 24 10 01 	movaps 0x110(%rsp),%xmm14
  402980:	00 00 
  402982:	44 0f 28 bc 24 20 01 	movaps 0x120(%rsp),%xmm15
  402989:	00 00 
  40298b:	48 81 c4 38 01 00 00 	add    $0x138,%rsp
  402992:	5b                   	pop    %rbx
  402993:	5d                   	pop    %rbp
  402994:	5f                   	pop    %rdi
  402995:	5e                   	pop    %rsi
  402996:	59                   	pop    %rcx
  402997:	5a                   	pop    %rdx
  402998:	41 58                	pop    %r8
  40299a:	41 59                	pop    %r9
  40299c:	41 5a                	pop    %r10
  40299e:	41 5b                	pop    %r11
  4029a0:	c3                   	ret
  4029a1:	48 89 e0             	mov    %rsp,%rax
  4029a4:	b9 05 00 00 00       	mov    $0x5,%ecx
  4029a9:	e8 92 11 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4029ae:	85 c0                	test   %eax,%eax
  4029b0:	0f 85 5c ff ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  4029b6:	41 f7 c2 00 00 00 02 	test   $0x2000000,%r10d
  4029bd:	74 15                	je     4029d4 <__intel_cpu_features_init_body+0x4f4>
  4029bf:	48 89 e0             	mov    %rsp,%rax
  4029c2:	b9 06 00 00 00       	mov    $0x6,%ecx
  4029c7:	e8 74 11 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4029cc:	85 c0                	test   %eax,%eax
  4029ce:	0f 85 3e ff ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  4029d4:	41 f7 c2 00 00 00 04 	test   $0x4000000,%r10d
  4029db:	74 15                	je     4029f2 <__intel_cpu_features_init_body+0x512>
  4029dd:	48 89 e0             	mov    %rsp,%rax
  4029e0:	b9 07 00 00 00       	mov    $0x7,%ecx
  4029e5:	e8 56 11 00 00       	call   403b40 <__libirc_set_cpu_feature>
  4029ea:	85 c0                	test   %eax,%eax
  4029ec:	0f 85 20 ff ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  4029f2:	41 f6 c0 01          	test   $0x1,%r8b
  4029f6:	74 15                	je     402a0d <__intel_cpu_features_init_body+0x52d>
  4029f8:	48 89 e0             	mov    %rsp,%rax
  4029fb:	b9 08 00 00 00       	mov    $0x8,%ecx
  402a00:	e8 3b 11 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402a05:	85 c0                	test   %eax,%eax
  402a07:	0f 85 05 ff ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402a0d:	41 f7 c0 00 02 00 00 	test   $0x200,%r8d
  402a14:	74 15                	je     402a2b <__intel_cpu_features_init_body+0x54b>
  402a16:	48 89 e0             	mov    %rsp,%rax
  402a19:	b9 09 00 00 00       	mov    $0x9,%ecx
  402a1e:	e8 1d 11 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402a23:	85 c0                	test   %eax,%eax
  402a25:	0f 85 e7 fe ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402a2b:	41 f7 c0 00 00 40 00 	test   $0x400000,%r8d
  402a32:	74 15                	je     402a49 <__intel_cpu_features_init_body+0x569>
  402a34:	48 89 e0             	mov    %rsp,%rax
  402a37:	b9 0c 00 00 00       	mov    $0xc,%ecx
  402a3c:	e8 ff 10 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402a41:	85 c0                	test   %eax,%eax
  402a43:	0f 85 c9 fe ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402a49:	41 f7 c0 00 00 08 00 	test   $0x80000,%r8d
  402a50:	74 15                	je     402a67 <__intel_cpu_features_init_body+0x587>
  402a52:	48 89 e0             	mov    %rsp,%rax
  402a55:	b9 0a 00 00 00       	mov    $0xa,%ecx
  402a5a:	e8 e1 10 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402a5f:	85 c0                	test   %eax,%eax
  402a61:	0f 85 ab fe ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402a67:	41 f7 c0 00 00 10 00 	test   $0x100000,%r8d
  402a6e:	74 15                	je     402a85 <__intel_cpu_features_init_body+0x5a5>
  402a70:	48 89 e0             	mov    %rsp,%rax
  402a73:	b9 0b 00 00 00       	mov    $0xb,%ecx
  402a78:	e8 c3 10 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402a7d:	85 c0                	test   %eax,%eax
  402a7f:	0f 85 8d fe ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402a85:	41 f7 c0 00 00 80 00 	test   $0x800000,%r8d
  402a8c:	74 15                	je     402aa3 <__intel_cpu_features_init_body+0x5c3>
  402a8e:	48 89 e0             	mov    %rsp,%rax
  402a91:	b9 0d 00 00 00       	mov    $0xd,%ecx
  402a96:	e8 a5 10 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402a9b:	85 c0                	test   %eax,%eax
  402a9d:	0f 85 6f fe ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402aa3:	41 f6 c0 02          	test   $0x2,%r8b
  402aa7:	74 15                	je     402abe <__intel_cpu_features_init_body+0x5de>
  402aa9:	48 89 e0             	mov    %rsp,%rax
  402aac:	b9 0e 00 00 00       	mov    $0xe,%ecx
  402ab1:	e8 8a 10 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402ab6:	85 c0                	test   %eax,%eax
  402ab8:	0f 85 54 fe ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402abe:	41 f7 c0 00 00 00 02 	test   $0x2000000,%r8d
  402ac5:	74 15                	je     402adc <__intel_cpu_features_init_body+0x5fc>
  402ac7:	48 89 e0             	mov    %rsp,%rax
  402aca:	b9 0f 00 00 00       	mov    $0xf,%ecx
  402acf:	e8 6c 10 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402ad4:	85 c0                	test   %eax,%eax
  402ad6:	0f 85 36 fe ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402adc:	b8 07 00 00 00       	mov    $0x7,%eax
  402ae1:	31 c9                	xor    %ecx,%ecx
  402ae3:	0f a2                	cpuid
  402ae5:	89 cf                	mov    %ecx,%edi
  402ae7:	89 d6                	mov    %edx,%esi
  402ae9:	41 89 d9             	mov    %ebx,%r9d
  402aec:	f7 c3 00 00 00 20    	test   $0x20000000,%ebx
  402af2:	0f 84 55 fb ff ff    	je     40264d <__intel_cpu_features_init_body+0x16d>
  402af8:	48 89 e0             	mov    %rsp,%rax
  402afb:	b9 24 00 00 00       	mov    $0x24,%ecx
  402b00:	e8 3b 10 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402b05:	85 c0                	test   %eax,%eax
  402b07:	0f 85 05 fe ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402b0d:	e9 3b fb ff ff       	jmp    40264d <__intel_cpu_features_init_body+0x16d>
  402b12:	48 89 e0             	mov    %rsp,%rax
  402b15:	b9 01 00 00 00       	mov    $0x1,%ecx
  402b1a:	e8 21 10 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402b1f:	85 c0                	test   %eax,%eax
  402b21:	0f 85 eb fd ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402b27:	b8 19 00 00 00       	mov    $0x19,%eax
  402b2c:	31 c9                	xor    %ecx,%ecx
  402b2e:	0f a2                	cpuid
  402b30:	f6 c3 01             	test   $0x1,%bl
  402b33:	74 15                	je     402b4a <__intel_cpu_features_init_body+0x66a>
  402b35:	48 89 e0             	mov    %rsp,%rax
  402b38:	b9 45 00 00 00       	mov    $0x45,%ecx
  402b3d:	e8 fe 0f 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402b42:	85 c0                	test   %eax,%eax
  402b44:	0f 85 c8 fd ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402b4a:	f6 c3 04             	test   $0x4,%bl
  402b4d:	0f 84 85 fd ff ff    	je     4028d8 <__intel_cpu_features_init_body+0x3f8>
  402b53:	48 89 e0             	mov    %rsp,%rax
  402b56:	b9 46 00 00 00       	mov    $0x46,%ecx
  402b5b:	e8 e0 0f 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402b60:	85 c0                	test   %eax,%eax
  402b62:	0f 85 aa fd ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402b68:	e9 6b fd ff ff       	jmp    4028d8 <__intel_cpu_features_init_body+0x3f8>
  402b6d:	48 89 e0             	mov    %rsp,%rax
  402b70:	b9 01 00 00 00       	mov    $0x1,%ecx
  402b75:	e8 c6 0f 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402b7a:	85 c0                	test   %eax,%eax
  402b7c:	0f 85 90 fd ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402b82:	31 c9                	xor    %ecx,%ecx
  402b84:	0f 01 d0             	xgetbv
  402b87:	41 89 c2             	mov    %eax,%r10d
  402b8a:	41 f7 d2             	not    %r10d
  402b8d:	41 f7 c2 00 00 06 00 	test   $0x60000,%r10d
  402b94:	75 6c                	jne    402c02 <__intel_cpu_features_init_body+0x722>
  402b96:	48 89 e0             	mov    %rsp,%rax
  402b99:	b9 01 00 00 00       	mov    $0x1,%ecx
  402b9e:	e8 9d 0f 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402ba3:	85 c0                	test   %eax,%eax
  402ba5:	0f 85 67 fd ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402bab:	f7 c6 00 00 00 01    	test   $0x1000000,%esi
  402bb1:	74 15                	je     402bc8 <__intel_cpu_features_init_body+0x6e8>
  402bb3:	48 89 e0             	mov    %rsp,%rax
  402bb6:	b9 42 00 00 00       	mov    $0x42,%ecx
  402bbb:	e8 80 0f 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402bc0:	85 c0                	test   %eax,%eax
  402bc2:	0f 85 4a fd ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402bc8:	f7 c6 00 00 00 02    	test   $0x2000000,%esi
  402bce:	74 15                	je     402be5 <__intel_cpu_features_init_body+0x705>
  402bd0:	48 89 e0             	mov    %rsp,%rax
  402bd3:	b9 43 00 00 00       	mov    $0x43,%ecx
  402bd8:	e8 63 0f 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402bdd:	85 c0                	test   %eax,%eax
  402bdf:	0f 85 2d fd ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402be5:	f7 c6 00 00 40 00    	test   $0x400000,%esi
  402beb:	74 15                	je     402c02 <__intel_cpu_features_init_body+0x722>
  402bed:	48 89 e0             	mov    %rsp,%rax
  402bf0:	b9 44 00 00 00       	mov    $0x44,%ecx
  402bf5:	e8 46 0f 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402bfa:	85 c0                	test   %eax,%eax
  402bfc:	0f 85 10 fd ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402c02:	41 f6 c2 06          	test   $0x6,%r10b
  402c06:	0f 85 d9 fc ff ff    	jne    4028e5 <__intel_cpu_features_init_body+0x405>
  402c0c:	48 89 e0             	mov    %rsp,%rax
  402c0f:	b9 01 00 00 00       	mov    $0x1,%ecx
  402c14:	e8 27 0f 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402c19:	85 c0                	test   %eax,%eax
  402c1b:	0f 85 f1 fc ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402c21:	41 f7 c0 00 00 00 10 	test   $0x10000000,%r8d
  402c28:	0f 85 f1 02 00 00    	jne    402f1f <__intel_cpu_features_init_body+0xa3f>
  402c2e:	41 f7 c0 00 00 00 20 	test   $0x20000000,%r8d
  402c35:	74 15                	je     402c4c <__intel_cpu_features_init_body+0x76c>
  402c37:	48 89 e0             	mov    %rsp,%rax
  402c3a:	b9 11 00 00 00       	mov    $0x11,%ecx
  402c3f:	e8 fc 0e 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402c44:	85 c0                	test   %eax,%eax
  402c46:	0f 85 c6 fc ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402c4c:	41 f6 c1 20          	test   $0x20,%r9b
  402c50:	74 15                	je     402c67 <__intel_cpu_features_init_body+0x787>
  402c52:	48 89 e0             	mov    %rsp,%rax
  402c55:	b9 18 00 00 00       	mov    $0x18,%ecx
  402c5a:	e8 e1 0e 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402c5f:	85 c0                	test   %eax,%eax
  402c61:	0f 85 ab fc ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402c67:	41 f7 c0 00 10 00 00 	test   $0x1000,%r8d
  402c6e:	74 15                	je     402c85 <__intel_cpu_features_init_body+0x7a5>
  402c70:	48 89 e0             	mov    %rsp,%rax
  402c73:	b9 13 00 00 00       	mov    $0x13,%ecx
  402c78:	e8 c3 0e 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402c7d:	85 c0                	test   %eax,%eax
  402c7f:	0f 85 8d fc ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402c85:	41 f6 c2 18          	test   $0x18,%r10b
  402c89:	75 33                	jne    402cbe <__intel_cpu_features_init_body+0x7de>
  402c8b:	48 89 e0             	mov    %rsp,%rax
  402c8e:	b9 01 00 00 00       	mov    $0x1,%ecx
  402c93:	e8 a8 0e 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402c98:	85 c0                	test   %eax,%eax
  402c9a:	0f 85 72 fc ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402ca0:	41 f7 c1 00 40 00 00 	test   $0x4000,%r9d
  402ca7:	74 15                	je     402cbe <__intel_cpu_features_init_body+0x7de>
  402ca9:	48 89 e0             	mov    %rsp,%rax
  402cac:	b9 25 00 00 00       	mov    $0x25,%ecx
  402cb1:	e8 8a 0e 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402cb6:	85 c0                	test   %eax,%eax
  402cb8:	0f 85 54 fc ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402cbe:	b8 07 00 00 00       	mov    $0x7,%eax
  402cc3:	b9 01 00 00 00       	mov    $0x1,%ecx
  402cc8:	0f a2                	cpuid
  402cca:	89 c2                	mov    %eax,%edx
  402ccc:	f6 c2 10             	test   $0x10,%dl
  402ccf:	74 15                	je     402ce6 <__intel_cpu_features_init_body+0x806>
  402cd1:	48 89 e0             	mov    %rsp,%rax
  402cd4:	b9 41 00 00 00       	mov    $0x41,%ecx
  402cd9:	e8 62 0e 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402cde:	85 c0                	test   %eax,%eax
  402ce0:	0f 85 2c fc ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402ce6:	41 f6 c2 e0          	test   $0xe0,%r10b
  402cea:	0f 85 f5 fb ff ff    	jne    4028e5 <__intel_cpu_features_init_body+0x405>
  402cf0:	48 89 e0             	mov    %rsp,%rax
  402cf3:	b9 01 00 00 00       	mov    $0x1,%ecx
  402cf8:	e8 43 0e 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402cfd:	85 c0                	test   %eax,%eax
  402cff:	0f 85 0d fc ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402d05:	41 f7 c1 00 00 01 00 	test   $0x10000,%r9d
  402d0c:	74 15                	je     402d23 <__intel_cpu_features_init_body+0x843>
  402d0e:	48 89 e0             	mov    %rsp,%rax
  402d11:	b9 19 00 00 00       	mov    $0x19,%ecx
  402d16:	e8 25 0e 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402d1b:	85 c0                	test   %eax,%eax
  402d1d:	0f 85 ef fb ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402d23:	41 f7 c1 00 00 00 10 	test   $0x10000000,%r9d
  402d2a:	74 15                	je     402d41 <__intel_cpu_features_init_body+0x861>
  402d2c:	48 89 e0             	mov    %rsp,%rax
  402d2f:	b9 23 00 00 00       	mov    $0x23,%ecx
  402d34:	e8 07 0e 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402d39:	85 c0                	test   %eax,%eax
  402d3b:	0f 85 d1 fb ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402d41:	41 f7 c1 00 00 00 08 	test   $0x8000000,%r9d
  402d48:	74 15                	je     402d5f <__intel_cpu_features_init_body+0x87f>
  402d4a:	48 89 e0             	mov    %rsp,%rax
  402d4d:	b9 21 00 00 00       	mov    $0x21,%ecx
  402d52:	e8 e9 0d 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402d57:	85 c0                	test   %eax,%eax
  402d59:	0f 85 b3 fb ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402d5f:	41 f7 c1 00 00 00 04 	test   $0x4000000,%r9d
  402d66:	74 15                	je     402d7d <__intel_cpu_features_init_body+0x89d>
  402d68:	48 89 e0             	mov    %rsp,%rax
  402d6b:	b9 22 00 00 00       	mov    $0x22,%ecx
  402d70:	e8 cb 0d 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402d75:	85 c0                	test   %eax,%eax
  402d77:	0f 85 95 fb ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402d7d:	41 f7 c1 00 00 02 00 	test   $0x20000,%r9d
  402d84:	74 15                	je     402d9b <__intel_cpu_features_init_body+0x8bb>
  402d86:	48 89 e0             	mov    %rsp,%rax
  402d89:	b9 1a 00 00 00       	mov    $0x1a,%ecx
  402d8e:	e8 ad 0d 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402d93:	85 c0                	test   %eax,%eax
  402d95:	0f 85 77 fb ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402d9b:	41 f7 c1 00 00 00 40 	test   $0x40000000,%r9d
  402da2:	74 15                	je     402db9 <__intel_cpu_features_init_body+0x8d9>
  402da4:	48 89 e0             	mov    %rsp,%rax
  402da7:	b9 26 00 00 00       	mov    $0x26,%ecx
  402dac:	e8 8f 0d 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402db1:	85 c0                	test   %eax,%eax
  402db3:	0f 85 59 fb ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402db9:	45 85 c9             	test   %r9d,%r9d
  402dbc:	0f 88 b5 01 00 00    	js     402f77 <__intel_cpu_features_init_body+0xa97>
  402dc2:	41 f7 c1 00 00 20 00 	test   $0x200000,%r9d
  402dc9:	74 15                	je     402de0 <__intel_cpu_features_init_body+0x900>
  402dcb:	48 89 e0             	mov    %rsp,%rax
  402dce:	b9 1f 00 00 00       	mov    $0x1f,%ecx
  402dd3:	e8 68 0d 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402dd8:	85 c0                	test   %eax,%eax
  402dda:	0f 85 32 fb ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402de0:	40 f6 c7 02          	test   $0x2,%dil
  402de4:	74 15                	je     402dfb <__intel_cpu_features_init_body+0x91b>
  402de6:	48 89 e0             	mov    %rsp,%rax
  402de9:	b9 28 00 00 00       	mov    $0x28,%ecx
  402dee:	e8 4d 0d 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402df3:	85 c0                	test   %eax,%eax
  402df5:	0f 85 17 fb ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402dfb:	f7 c7 00 40 00 00    	test   $0x4000,%edi
  402e01:	74 15                	je     402e18 <__intel_cpu_features_init_body+0x938>
  402e03:	48 89 e0             	mov    %rsp,%rax
  402e06:	b9 2b 00 00 00       	mov    $0x2b,%ecx
  402e0b:	e8 30 0d 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402e10:	85 c0                	test   %eax,%eax
  402e12:	0f 85 fa fa ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402e18:	40 f6 c6 04          	test   $0x4,%sil
  402e1c:	74 15                	je     402e33 <__intel_cpu_features_init_body+0x953>
  402e1e:	48 89 e0             	mov    %rsp,%rax
  402e21:	b9 2a 00 00 00       	mov    $0x2a,%ecx
  402e26:	e8 15 0d 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402e2b:	85 c0                	test   %eax,%eax
  402e2d:	0f 85 df fa ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402e33:	40 f6 c6 08          	test   $0x8,%sil
  402e37:	74 15                	je     402e4e <__intel_cpu_features_init_body+0x96e>
  402e39:	48 89 e0             	mov    %rsp,%rax
  402e3c:	b9 29 00 00 00       	mov    $0x29,%ecx
  402e41:	e8 fa 0c 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402e46:	85 c0                	test   %eax,%eax
  402e48:	0f 85 c4 fa ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402e4e:	f7 c7 00 10 00 00    	test   $0x1000,%edi
  402e54:	74 15                	je     402e6b <__intel_cpu_features_init_body+0x98b>
  402e56:	48 89 e0             	mov    %rsp,%rax
  402e59:	b9 2c 00 00 00       	mov    $0x2c,%ecx
  402e5e:	e8 dd 0c 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402e63:	85 c0                	test   %eax,%eax
  402e65:	0f 85 a7 fa ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402e6b:	40 f6 c7 40          	test   $0x40,%dil
  402e6f:	74 15                	je     402e86 <__intel_cpu_features_init_body+0x9a6>
  402e71:	48 89 e0             	mov    %rsp,%rax
  402e74:	b9 2d 00 00 00       	mov    $0x2d,%ecx
  402e79:	e8 c2 0c 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402e7e:	85 c0                	test   %eax,%eax
  402e80:	0f 85 8c fa ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402e86:	f7 c7 00 08 00 00    	test   $0x800,%edi
  402e8c:	74 15                	je     402ea3 <__intel_cpu_features_init_body+0x9c3>
  402e8e:	48 89 e0             	mov    %rsp,%rax
  402e91:	b9 31 00 00 00       	mov    $0x31,%ecx
  402e96:	e8 a5 0c 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402e9b:	85 c0                	test   %eax,%eax
  402e9d:	0f 85 6f fa ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402ea3:	f6 c2 20             	test   $0x20,%dl
  402ea6:	74 15                	je     402ebd <__intel_cpu_features_init_body+0x9dd>
  402ea8:	48 89 e0             	mov    %rsp,%rax
  402eab:	b9 3f 00 00 00       	mov    $0x3f,%ecx
  402eb0:	e8 8b 0c 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402eb5:	85 c0                	test   %eax,%eax
  402eb7:	0f 85 55 fa ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402ebd:	f7 c6 00 00 80 00    	test   $0x800000,%esi
  402ec3:	74 15                	je     402eda <__intel_cpu_features_init_body+0x9fa>
  402ec5:	48 89 e0             	mov    %rsp,%rax
  402ec8:	b9 3a 00 00 00       	mov    $0x3a,%ecx
  402ecd:	e8 6e 0c 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402ed2:	85 c0                	test   %eax,%eax
  402ed4:	0f 85 38 fa ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402eda:	f7 c6 00 01 00 00    	test   $0x100,%esi
  402ee0:	0f 84 ff f9 ff ff    	je     4028e5 <__intel_cpu_features_init_body+0x405>
  402ee6:	48 89 e0             	mov    %rsp,%rax
  402ee9:	b9 39 00 00 00       	mov    $0x39,%ecx
  402eee:	e8 4d 0c 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402ef3:	85 c0                	test   %eax,%eax
  402ef5:	0f 85 17 fa ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402efb:	e9 e5 f9 ff ff       	jmp    4028e5 <__intel_cpu_features_init_body+0x405>
  402f00:	0f 28 04 24          	movaps (%rsp),%xmm0
  402f04:	83 fd 01             	cmp    $0x1,%ebp
  402f07:	75 07                	jne    402f10 <__intel_cpu_features_init_body+0xa30>
  402f09:	0f 29 05 f0 51 00 00 	movaps %xmm0,0x51f0(%rip)        # 408100 <__intel_cpu_feature_indicator>
  402f10:	48 c7 c0 10 81 40 00 	mov    $0x408110,%rax
  402f17:	0f 29 00             	movaps %xmm0,(%rax)
  402f1a:	e9 f1 f9 ff ff       	jmp    402910 <__intel_cpu_features_init_body+0x430>
  402f1f:	48 89 e0             	mov    %rsp,%rax
  402f22:	b9 10 00 00 00       	mov    $0x10,%ecx
  402f27:	e8 14 0c 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402f2c:	85 c0                	test   %eax,%eax
  402f2e:	0f 85 de f9 ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402f34:	f7 c7 00 02 00 00    	test   $0x200,%edi
  402f3a:	74 15                	je     402f51 <__intel_cpu_features_init_body+0xa71>
  402f3c:	48 89 e0             	mov    %rsp,%rax
  402f3f:	b9 2f 00 00 00       	mov    $0x2f,%ecx
  402f44:	e8 f7 0b 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402f49:	85 c0                	test   %eax,%eax
  402f4b:	0f 85 c1 f9 ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402f51:	f7 c7 00 04 00 00    	test   $0x400,%edi
  402f57:	0f 84 d1 fc ff ff    	je     402c2e <__intel_cpu_features_init_body+0x74e>
  402f5d:	48 89 e0             	mov    %rsp,%rax
  402f60:	b9 30 00 00 00       	mov    $0x30,%ecx
  402f65:	e8 d6 0b 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402f6a:	85 c0                	test   %eax,%eax
  402f6c:	0f 85 a0 f9 ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402f72:	e9 b7 fc ff ff       	jmp    402c2e <__intel_cpu_features_init_body+0x74e>
  402f77:	48 89 e0             	mov    %rsp,%rax
  402f7a:	b9 27 00 00 00       	mov    $0x27,%ecx
  402f7f:	e8 bc 0b 00 00       	call   403b40 <__libirc_set_cpu_feature>
  402f84:	85 c0                	test   %eax,%eax
  402f86:	0f 85 86 f9 ff ff    	jne    402912 <__intel_cpu_features_init_body+0x432>
  402f8c:	e9 31 fe ff ff       	jmp    402dc2 <__intel_cpu_features_init_body+0x8e2>
  402f91:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  402f98:	00 00 00 
  402f9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000402fa0 <__intel_cpu_features_init_x>:
  402fa0:	f3 0f 1e fa          	endbr64
  402fa4:	50                   	push   %rax
  402fa5:	31 c0                	xor    %eax,%eax
  402fa7:	e8 34 f5 ff ff       	call   4024e0 <__intel_cpu_features_init_body>
  402fac:	48 83 c4 08          	add    $0x8,%rsp
  402fb0:	c3                   	ret
  402fb1:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  402fb8:	00 00 00 
  402fbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000402fc0 <__libirc_get_feature_name>:
  402fc0:	f3 0f 1e fa          	endbr64
  402fc4:	50                   	push   %rax
  402fc5:	80 3d 54 51 00 00 00 	cmpb   $0x0,0x5154(%rip)        # 408120 <__libirc_isa_info_initialized>
  402fcc:	75 05                	jne    402fd3 <__libirc_get_feature_name+0x13>
  402fce:	e8 1d 00 00 00       	call   402ff0 <__libirc_isa_init_once>
  402fd3:	89 f8                	mov    %edi,%eax
  402fd5:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  402fd9:	48 8d 0d 50 51 00 00 	lea    0x5150(%rip),%rcx        # 408130 <proc_info_features>
  402fe0:	48 8b 04 c1          	mov    (%rcx,%rax,8),%rax
  402fe4:	59                   	pop    %rcx
  402fe5:	c3                   	ret
  402fe6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  402fed:	00 00 00 

0000000000402ff0 <__libirc_isa_init_once>:
  402ff0:	51                   	push   %rcx
  402ff1:	80 3d 28 51 00 00 00 	cmpb   $0x0,0x5128(%rip)        # 408120 <__libirc_isa_info_initialized>
  402ff8:	0f 85 aa 0a 00 00    	jne    403aa8 <__libirc_isa_init_once+0xab8>
  402ffe:	b8 c8 00 00 00       	mov    $0xc8,%eax
  403003:	48 8d 0d 26 51 00 00 	lea    0x5126(%rip),%rcx        # 408130 <proc_info_features>
  40300a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403010:	c7 84 08 58 ff ff ff 	movl   $0xffffffff,-0xa8(%rax,%rcx,1)
  403017:	ff ff ff ff 
  40301b:	c7 84 08 70 ff ff ff 	movl   $0xffffffff,-0x90(%rax,%rcx,1)
  403022:	ff ff ff ff 
  403026:	c7 44 08 88 ff ff ff 	movl   $0xffffffff,-0x78(%rax,%rcx,1)
  40302d:	ff 
  40302e:	c7 44 08 a0 ff ff ff 	movl   $0xffffffff,-0x60(%rax,%rcx,1)
  403035:	ff 
  403036:	c7 44 08 b8 ff ff ff 	movl   $0xffffffff,-0x48(%rax,%rcx,1)
  40303d:	ff 
  40303e:	c7 44 08 d0 ff ff ff 	movl   $0xffffffff,-0x30(%rax,%rcx,1)
  403045:	ff 
  403046:	c7 44 08 e8 ff ff ff 	movl   $0xffffffff,-0x18(%rax,%rcx,1)
  40304d:	ff 
  40304e:	c7 04 08 ff ff ff ff 	movl   $0xffffffff,(%rax,%rcx,1)
  403055:	48 05 c0 00 00 00    	add    $0xc0,%rax
  40305b:	48 3d c8 06 00 00    	cmp    $0x6c8,%rax
  403061:	75 ad                	jne    403010 <__libirc_isa_init_once+0x20>
  403063:	c7 05 e3 56 00 00 ff 	movl   $0xffffffff,0x56e3(%rip)        # 408750 <proc_info_features+0x620>
  40306a:	ff ff ff 
  40306d:	c7 05 f1 56 00 00 ff 	movl   $0xffffffff,0x56f1(%rip)        # 408768 <proc_info_features+0x638>
  403074:	ff ff ff 
  403077:	c7 05 ff 56 00 00 ff 	movl   $0xffffffff,0x56ff(%rip)        # 408780 <proc_info_features+0x650>
  40307e:	ff ff ff 
  403081:	c7 05 0d 57 00 00 ff 	movl   $0xffffffff,0x570d(%rip)        # 408798 <proc_info_features+0x668>
  403088:	ff ff ff 
  40308b:	c7 05 1b 57 00 00 ff 	movl   $0xffffffff,0x571b(%rip)        # 4087b0 <proc_info_features+0x680>
  403092:	ff ff ff 
  403095:	c7 05 29 57 00 00 ff 	movl   $0xffffffff,0x5729(%rip)        # 4087c8 <proc_info_features+0x698>
  40309c:	ff ff ff 
  40309f:	48 8d 05 c7 1f 00 00 	lea    0x1fc7(%rip),%rax        # 40506d <_IO_stdin_used+0x6d>
  4030a6:	48 89 05 9b 50 00 00 	mov    %rax,0x509b(%rip)        # 408148 <proc_info_features+0x18>
  4030ad:	c7 05 99 50 00 00 00 	movl   $0x0,0x5099(%rip)        # 408150 <proc_info_features+0x20>
  4030b4:	00 00 00 
  4030b7:	48 8d 05 bc 1f 00 00 	lea    0x1fbc(%rip),%rax        # 40507a <_IO_stdin_used+0x7a>
  4030be:	48 89 05 9b 50 00 00 	mov    %rax,0x509b(%rip)        # 408160 <proc_info_features+0x30>
  4030c5:	c7 05 99 50 00 00 01 	movl   $0x1,0x5099(%rip)        # 408168 <proc_info_features+0x38>
  4030cc:	00 00 00 
  4030cf:	48 8d 05 a8 1f 00 00 	lea    0x1fa8(%rip),%rax        # 40507e <_IO_stdin_used+0x7e>
  4030d6:	48 89 05 9b 50 00 00 	mov    %rax,0x509b(%rip)        # 408178 <proc_info_features+0x48>
  4030dd:	c7 05 99 50 00 00 02 	movl   $0x2,0x5099(%rip)        # 408180 <proc_info_features+0x50>
  4030e4:	00 00 00 
  4030e7:	c7 05 a7 50 00 00 03 	movl   $0x3,0x50a7(%rip)        # 408198 <proc_info_features+0x68>
  4030ee:	00 00 00 
  4030f1:	48 8d 05 8b 1f 00 00 	lea    0x1f8b(%rip),%rax        # 405083 <_IO_stdin_used+0x83>
  4030f8:	48 89 05 a1 50 00 00 	mov    %rax,0x50a1(%rip)        # 4081a0 <proc_info_features+0x70>
  4030ff:	48 8d 05 81 1f 00 00 	lea    0x1f81(%rip),%rax        # 405087 <_IO_stdin_used+0x87>
  403106:	48 89 05 83 50 00 00 	mov    %rax,0x5083(%rip)        # 408190 <proc_info_features+0x60>
  40310d:	48 8d 05 77 1f 00 00 	lea    0x1f77(%rip),%rax        # 40508b <_IO_stdin_used+0x8b>
  403114:	48 89 05 8d 50 00 00 	mov    %rax,0x508d(%rip)        # 4081a8 <proc_info_features+0x78>
  40311b:	c7 05 8b 50 00 00 04 	movl   $0x4,0x508b(%rip)        # 4081b0 <proc_info_features+0x80>
  403122:	00 00 00 
  403125:	c7 05 99 50 00 00 05 	movl   $0x5,0x5099(%rip)        # 4081c8 <proc_info_features+0x98>
  40312c:	00 00 00 
  40312f:	48 8d 05 5c 1f 00 00 	lea    0x1f5c(%rip),%rax        # 405092 <_IO_stdin_used+0x92>
  403136:	48 89 05 93 50 00 00 	mov    %rax,0x5093(%rip)        # 4081d0 <proc_info_features+0xa0>
  40313d:	48 8d 05 52 1f 00 00 	lea    0x1f52(%rip),%rax        # 405096 <_IO_stdin_used+0x96>
  403144:	48 89 05 75 50 00 00 	mov    %rax,0x5075(%rip)        # 4081c0 <proc_info_features+0x90>
  40314b:	c7 05 8b 50 00 00 06 	movl   $0x6,0x508b(%rip)        # 4081e0 <proc_info_features+0xb0>
  403152:	00 00 00 
  403155:	48 8d 05 3e 1f 00 00 	lea    0x1f3e(%rip),%rax        # 40509a <_IO_stdin_used+0x9a>
  40315c:	48 89 05 85 50 00 00 	mov    %rax,0x5085(%rip)        # 4081e8 <proc_info_features+0xb8>
  403163:	48 8d 05 35 1f 00 00 	lea    0x1f35(%rip),%rax        # 40509f <_IO_stdin_used+0x9f>
  40316a:	48 89 05 67 50 00 00 	mov    %rax,0x5067(%rip)        # 4081d8 <proc_info_features+0xa8>
  403171:	c7 05 7d 50 00 00 07 	movl   $0x7,0x507d(%rip)        # 4081f8 <proc_info_features+0xc8>
  403178:	00 00 00 
  40317b:	48 8d 05 23 1f 00 00 	lea    0x1f23(%rip),%rax        # 4050a5 <_IO_stdin_used+0xa5>
  403182:	48 89 05 77 50 00 00 	mov    %rax,0x5077(%rip)        # 408200 <proc_info_features+0xd0>
  403189:	48 8d 05 1b 1f 00 00 	lea    0x1f1b(%rip),%rax        # 4050ab <_IO_stdin_used+0xab>
  403190:	48 89 05 59 50 00 00 	mov    %rax,0x5059(%rip)        # 4081f0 <proc_info_features+0xc0>
  403197:	c7 05 6f 50 00 00 08 	movl   $0x8,0x506f(%rip)        # 408210 <proc_info_features+0xe0>
  40319e:	00 00 00 
  4031a1:	48 8d 05 fc 1e 00 00 	lea    0x1efc(%rip),%rax        # 4050a4 <_IO_stdin_used+0xa4>
  4031a8:	48 89 05 69 50 00 00 	mov    %rax,0x5069(%rip)        # 408218 <proc_info_features+0xe8>
  4031af:	48 8d 05 f4 1e 00 00 	lea    0x1ef4(%rip),%rax        # 4050aa <_IO_stdin_used+0xaa>
  4031b6:	48 89 05 4b 50 00 00 	mov    %rax,0x504b(%rip)        # 408208 <proc_info_features+0xd8>
  4031bd:	c7 05 61 50 00 00 09 	movl   $0x9,0x5061(%rip)        # 408228 <proc_info_features+0xf8>
  4031c4:	00 00 00 
  4031c7:	48 8d 05 e2 1e 00 00 	lea    0x1ee2(%rip),%rax        # 4050b0 <_IO_stdin_used+0xb0>
  4031ce:	48 89 05 5b 50 00 00 	mov    %rax,0x505b(%rip)        # 408230 <proc_info_features+0x100>
  4031d5:	48 8d 05 db 1e 00 00 	lea    0x1edb(%rip),%rax        # 4050b7 <_IO_stdin_used+0xb7>
  4031dc:	48 89 05 3d 50 00 00 	mov    %rax,0x503d(%rip)        # 408220 <proc_info_features+0xf0>
  4031e3:	c7 05 53 50 00 00 0a 	movl   $0xa,0x5053(%rip)        # 408240 <proc_info_features+0x110>
  4031ea:	00 00 00 
  4031ed:	48 8d 05 ca 1e 00 00 	lea    0x1eca(%rip),%rax        # 4050be <_IO_stdin_used+0xbe>
  4031f4:	48 89 05 4d 50 00 00 	mov    %rax,0x504d(%rip)        # 408248 <proc_info_features+0x118>
  4031fb:	48 8d 05 c1 1e 00 00 	lea    0x1ec1(%rip),%rax        # 4050c3 <_IO_stdin_used+0xc3>
  403202:	48 89 05 2f 50 00 00 	mov    %rax,0x502f(%rip)        # 408238 <proc_info_features+0x108>
  403209:	c7 05 45 50 00 00 0b 	movl   $0xb,0x5045(%rip)        # 408258 <proc_info_features+0x128>
  403210:	00 00 00 
  403213:	48 8d 05 b0 1e 00 00 	lea    0x1eb0(%rip),%rax        # 4050ca <_IO_stdin_used+0xca>
  40321a:	48 89 05 3f 50 00 00 	mov    %rax,0x503f(%rip)        # 408260 <proc_info_features+0x130>
  403221:	48 8d 05 a8 1e 00 00 	lea    0x1ea8(%rip),%rax        # 4050d0 <_IO_stdin_used+0xd0>
  403228:	48 89 05 21 50 00 00 	mov    %rax,0x5021(%rip)        # 408250 <proc_info_features+0x120>
  40322f:	c7 05 37 50 00 00 0c 	movl   $0xc,0x5037(%rip)        # 408270 <proc_info_features+0x140>
  403236:	00 00 00 
  403239:	48 8d 05 96 1e 00 00 	lea    0x1e96(%rip),%rax        # 4050d6 <_IO_stdin_used+0xd6>
  403240:	48 89 05 31 50 00 00 	mov    %rax,0x5031(%rip)        # 408278 <proc_info_features+0x148>
  403247:	48 8d 05 8f 1e 00 00 	lea    0x1e8f(%rip),%rax        # 4050dd <_IO_stdin_used+0xdd>
  40324e:	48 89 05 13 50 00 00 	mov    %rax,0x5013(%rip)        # 408268 <proc_info_features+0x138>
  403255:	c7 05 29 50 00 00 0d 	movl   $0xd,0x5029(%rip)        # 408288 <proc_info_features+0x158>
  40325c:	00 00 00 
  40325f:	48 8d 05 7e 1e 00 00 	lea    0x1e7e(%rip),%rax        # 4050e4 <_IO_stdin_used+0xe4>
  403266:	48 89 05 23 50 00 00 	mov    %rax,0x5023(%rip)        # 408290 <proc_info_features+0x160>
  40326d:	48 8d 05 50 20 00 00 	lea    0x2050(%rip),%rax        # 4052c4 <_IO_stdin_used+0x2c4>
  403274:	48 89 05 05 50 00 00 	mov    %rax,0x5005(%rip)        # 408280 <proc_info_features+0x150>
  40327b:	c7 05 1b 50 00 00 0e 	movl   $0xe,0x501b(%rip)        # 4082a0 <proc_info_features+0x170>
  403282:	00 00 00 
  403285:	48 8d 05 23 20 00 00 	lea    0x2023(%rip),%rax        # 4052af <_IO_stdin_used+0x2af>
  40328c:	48 89 05 15 50 00 00 	mov    %rax,0x5015(%rip)        # 4082a8 <proc_info_features+0x178>
  403293:	48 8d 05 1a 20 00 00 	lea    0x201a(%rip),%rax        # 4052b4 <_IO_stdin_used+0x2b4>
  40329a:	48 89 05 f7 4f 00 00 	mov    %rax,0x4ff7(%rip)        # 408298 <proc_info_features+0x168>
  4032a1:	c7 05 0d 50 00 00 10 	movl   $0x10,0x500d(%rip)        # 4082b8 <proc_info_features+0x188>
  4032a8:	00 00 00 
  4032ab:	48 8d 05 39 1e 00 00 	lea    0x1e39(%rip),%rax        # 4050eb <_IO_stdin_used+0xeb>
  4032b2:	48 89 05 07 50 00 00 	mov    %rax,0x5007(%rip)        # 4082c0 <proc_info_features+0x190>
  4032b9:	48 8d 05 2f 1e 00 00 	lea    0x1e2f(%rip),%rax        # 4050ef <_IO_stdin_used+0xef>
  4032c0:	48 89 05 e9 4f 00 00 	mov    %rax,0x4fe9(%rip)        # 4082b0 <proc_info_features+0x180>
  4032c7:	c7 05 ff 4f 00 00 0f 	movl   $0xf,0x4fff(%rip)        # 4082d0 <proc_info_features+0x1a0>
  4032ce:	00 00 00 
  4032d1:	48 8d 05 1b 1e 00 00 	lea    0x1e1b(%rip),%rax        # 4050f3 <_IO_stdin_used+0xf3>
  4032d8:	48 89 05 f9 4f 00 00 	mov    %rax,0x4ff9(%rip)        # 4082d8 <proc_info_features+0x1a8>
  4032df:	48 8d 05 12 1e 00 00 	lea    0x1e12(%rip),%rax        # 4050f8 <_IO_stdin_used+0xf8>
  4032e6:	48 89 05 db 4f 00 00 	mov    %rax,0x4fdb(%rip)        # 4082c8 <proc_info_features+0x198>
  4032ed:	c7 05 f1 4f 00 00 11 	movl   $0x11,0x4ff1(%rip)        # 4082e8 <proc_info_features+0x1b8>
  4032f4:	00 00 00 
  4032f7:	48 8d 05 ff 1d 00 00 	lea    0x1dff(%rip),%rax        # 4050fd <_IO_stdin_used+0xfd>
  4032fe:	48 89 05 eb 4f 00 00 	mov    %rax,0x4feb(%rip)        # 4082f0 <proc_info_features+0x1c0>
  403305:	48 8d 05 f7 1d 00 00 	lea    0x1df7(%rip),%rax        # 405103 <_IO_stdin_used+0x103>
  40330c:	48 89 05 cd 4f 00 00 	mov    %rax,0x4fcd(%rip)        # 4082e0 <proc_info_features+0x1b0>
  403313:	c7 05 e3 4f 00 00 12 	movl   $0x12,0x4fe3(%rip)        # 408300 <proc_info_features+0x1d0>
  40331a:	00 00 00 
  40331d:	48 8d 05 64 1e 00 00 	lea    0x1e64(%rip),%rax        # 405188 <_IO_stdin_used+0x188>
  403324:	48 89 05 dd 4f 00 00 	mov    %rax,0x4fdd(%rip)        # 408308 <proc_info_features+0x1d8>
  40332b:	48 8d 05 d7 1d 00 00 	lea    0x1dd7(%rip),%rax        # 405109 <_IO_stdin_used+0x109>
  403332:	48 89 05 bf 4f 00 00 	mov    %rax,0x4fbf(%rip)        # 4082f8 <proc_info_features+0x1c8>
  403339:	c7 05 d5 4f 00 00 13 	movl   $0x13,0x4fd5(%rip)        # 408318 <proc_info_features+0x1e8>
  403340:	00 00 00 
  403343:	48 8d 05 c0 1e 00 00 	lea    0x1ec0(%rip),%rax        # 40520a <_IO_stdin_used+0x20a>
  40334a:	48 89 05 cf 4f 00 00 	mov    %rax,0x4fcf(%rip)        # 408320 <proc_info_features+0x1f0>
  403351:	48 8d 05 bd 1e 00 00 	lea    0x1ebd(%rip),%rax        # 405215 <_IO_stdin_used+0x215>
  403358:	48 89 05 b1 4f 00 00 	mov    %rax,0x4fb1(%rip)        # 408310 <proc_info_features+0x1e0>
  40335f:	c7 05 c7 4f 00 00 14 	movl   $0x14,0x4fc7(%rip)        # 408330 <proc_info_features+0x200>
  403366:	00 00 00 
  403369:	48 8d 05 9d 1d 00 00 	lea    0x1d9d(%rip),%rax        # 40510d <_IO_stdin_used+0x10d>
  403370:	48 89 05 c1 4f 00 00 	mov    %rax,0x4fc1(%rip)        # 408338 <proc_info_features+0x208>
  403377:	48 8d 05 95 1d 00 00 	lea    0x1d95(%rip),%rax        # 405113 <_IO_stdin_used+0x113>
  40337e:	48 89 05 a3 4f 00 00 	mov    %rax,0x4fa3(%rip)        # 408328 <proc_info_features+0x1f8>
  403385:	c7 05 b9 4f 00 00 15 	movl   $0x15,0x4fb9(%rip)        # 408348 <proc_info_features+0x218>
  40338c:	00 00 00 
  40338f:	48 8d 05 83 1d 00 00 	lea    0x1d83(%rip),%rax        # 405119 <_IO_stdin_used+0x119>
  403396:	48 89 05 b3 4f 00 00 	mov    %rax,0x4fb3(%rip)        # 408350 <proc_info_features+0x220>
  40339d:	48 8d 05 79 1d 00 00 	lea    0x1d79(%rip),%rax        # 40511d <_IO_stdin_used+0x11d>
  4033a4:	48 89 05 95 4f 00 00 	mov    %rax,0x4f95(%rip)        # 408340 <proc_info_features+0x210>
  4033ab:	c7 05 ab 4f 00 00 16 	movl   $0x16,0x4fab(%rip)        # 408360 <proc_info_features+0x230>
  4033b2:	00 00 00 
  4033b5:	48 8d 05 65 1d 00 00 	lea    0x1d65(%rip),%rax        # 405121 <_IO_stdin_used+0x121>
  4033bc:	48 89 05 a5 4f 00 00 	mov    %rax,0x4fa5(%rip)        # 408368 <proc_info_features+0x238>
  4033c3:	48 8d 05 5b 1d 00 00 	lea    0x1d5b(%rip),%rax        # 405125 <_IO_stdin_used+0x125>
  4033ca:	48 89 05 87 4f 00 00 	mov    %rax,0x4f87(%rip)        # 408358 <proc_info_features+0x228>
  4033d1:	c7 05 9d 4f 00 00 17 	movl   $0x17,0x4f9d(%rip)        # 408378 <proc_info_features+0x248>
  4033d8:	00 00 00 
  4033db:	48 8d 05 47 1d 00 00 	lea    0x1d47(%rip),%rax        # 405129 <_IO_stdin_used+0x129>
  4033e2:	48 89 05 97 4f 00 00 	mov    %rax,0x4f97(%rip)        # 408380 <proc_info_features+0x250>
  4033e9:	48 8d 05 3e 1d 00 00 	lea    0x1d3e(%rip),%rax        # 40512e <_IO_stdin_used+0x12e>
  4033f0:	48 89 05 79 4f 00 00 	mov    %rax,0x4f79(%rip)        # 408370 <proc_info_features+0x240>
  4033f7:	c7 05 8f 4f 00 00 1b 	movl   $0x1b,0x4f8f(%rip)        # 408390 <proc_info_features+0x260>
  4033fe:	00 00 00 
  403401:	48 8d 05 2b 1d 00 00 	lea    0x1d2b(%rip),%rax        # 405133 <_IO_stdin_used+0x133>
  403408:	48 89 05 89 4f 00 00 	mov    %rax,0x4f89(%rip)        # 408398 <proc_info_features+0x268>
  40340f:	48 8d 05 25 1d 00 00 	lea    0x1d25(%rip),%rax        # 40513b <_IO_stdin_used+0x13b>
  403416:	48 89 05 6b 4f 00 00 	mov    %rax,0x4f6b(%rip)        # 408388 <proc_info_features+0x258>
  40341d:	c7 05 81 4f 00 00 18 	movl   $0x18,0x4f81(%rip)        # 4083a8 <proc_info_features+0x278>
  403424:	00 00 00 
  403427:	48 8d 05 15 1d 00 00 	lea    0x1d15(%rip),%rax        # 405143 <_IO_stdin_used+0x143>
  40342e:	48 89 05 7b 4f 00 00 	mov    %rax,0x4f7b(%rip)        # 4083b0 <proc_info_features+0x280>
  403435:	48 8d 05 10 1d 00 00 	lea    0x1d10(%rip),%rax        # 40514c <_IO_stdin_used+0x14c>
  40343c:	48 89 05 5d 4f 00 00 	mov    %rax,0x4f5d(%rip)        # 4083a0 <proc_info_features+0x270>
  403443:	c7 05 73 4f 00 00 19 	movl   $0x19,0x4f73(%rip)        # 4083c0 <proc_info_features+0x290>
  40344a:	00 00 00 
  40344d:	48 8d 05 01 1d 00 00 	lea    0x1d01(%rip),%rax        # 405155 <_IO_stdin_used+0x155>
  403454:	48 89 05 6d 4f 00 00 	mov    %rax,0x4f6d(%rip)        # 4083c8 <proc_info_features+0x298>
  40345b:	48 8d 05 fb 1c 00 00 	lea    0x1cfb(%rip),%rax        # 40515d <_IO_stdin_used+0x15d>
  403462:	48 89 05 4f 4f 00 00 	mov    %rax,0x4f4f(%rip)        # 4083b8 <proc_info_features+0x288>
  403469:	48 8d 05 f5 1c 00 00 	lea    0x1cf5(%rip),%rax        # 405165 <_IO_stdin_used+0x165>
  403470:	48 89 05 59 4f 00 00 	mov    %rax,0x4f59(%rip)        # 4083d0 <proc_info_features+0x2a0>
  403477:	c7 05 57 4f 00 00 1a 	movl   $0x1a,0x4f57(%rip)        # 4083d8 <proc_info_features+0x2a8>
  40347e:	00 00 00 
  403481:	c7 05 65 4f 00 00 1c 	movl   $0x1c,0x4f65(%rip)        # 4083f0 <proc_info_features+0x2c0>
  403488:	00 00 00 
  40348b:	48 8d 05 d9 1c 00 00 	lea    0x1cd9(%rip),%rax        # 40516b <_IO_stdin_used+0x16b>
  403492:	48 89 05 5f 4f 00 00 	mov    %rax,0x4f5f(%rip)        # 4083f8 <proc_info_features+0x2c8>
  403499:	48 8d 05 cf 1c 00 00 	lea    0x1ccf(%rip),%rax        # 40516f <_IO_stdin_used+0x16f>
  4034a0:	48 89 05 41 4f 00 00 	mov    %rax,0x4f41(%rip)        # 4083e8 <proc_info_features+0x2b8>
  4034a7:	c7 05 57 4f 00 00 1d 	movl   $0x1d,0x4f57(%rip)        # 408408 <proc_info_features+0x2d8>
  4034ae:	00 00 00 
  4034b1:	48 8d 05 bb 1c 00 00 	lea    0x1cbb(%rip),%rax        # 405173 <_IO_stdin_used+0x173>
  4034b8:	48 89 05 51 4f 00 00 	mov    %rax,0x4f51(%rip)        # 408410 <proc_info_features+0x2e0>
  4034bf:	48 8d 05 b4 1c 00 00 	lea    0x1cb4(%rip),%rax        # 40517a <_IO_stdin_used+0x17a>
  4034c6:	48 89 05 33 4f 00 00 	mov    %rax,0x4f33(%rip)        # 408400 <proc_info_features+0x2d0>
  4034cd:	c7 05 49 4f 00 00 1e 	movl   $0x1e,0x4f49(%rip)        # 408420 <proc_info_features+0x2f0>
  4034d4:	00 00 00 
  4034d7:	48 8d 05 a3 1c 00 00 	lea    0x1ca3(%rip),%rax        # 405181 <_IO_stdin_used+0x181>
  4034de:	48 89 05 43 4f 00 00 	mov    %rax,0x4f43(%rip)        # 408428 <proc_info_features+0x2f8>
  4034e5:	48 8d 05 a0 1c 00 00 	lea    0x1ca0(%rip),%rax        # 40518c <_IO_stdin_used+0x18c>
  4034ec:	48 89 05 25 4f 00 00 	mov    %rax,0x4f25(%rip)        # 408418 <proc_info_features+0x2e8>
  4034f3:	c7 05 3b 4f 00 00 ff 	movl   $0xffffffff,0x4f3b(%rip)        # 408438 <proc_info_features+0x308>
  4034fa:	ff ff ff 
  4034fd:	c7 05 49 4f 00 00 20 	movl   $0x20,0x4f49(%rip)        # 408450 <proc_info_features+0x320>
  403504:	00 00 00 
  403507:	48 8d 05 8b 1c 00 00 	lea    0x1c8b(%rip),%rax        # 405199 <_IO_stdin_used+0x199>
  40350e:	48 89 05 43 4f 00 00 	mov    %rax,0x4f43(%rip)        # 408458 <proc_info_features+0x328>
  403515:	48 8d 05 86 1c 00 00 	lea    0x1c86(%rip),%rax        # 4051a2 <_IO_stdin_used+0x1a2>
  40351c:	48 89 05 25 4f 00 00 	mov    %rax,0x4f25(%rip)        # 408448 <proc_info_features+0x318>
  403523:	c7 05 3b 4f 00 00 21 	movl   $0x21,0x4f3b(%rip)        # 408468 <proc_info_features+0x338>
  40352a:	00 00 00 
  40352d:	48 8d 05 77 1c 00 00 	lea    0x1c77(%rip),%rax        # 4051ab <_IO_stdin_used+0x1ab>
  403534:	48 89 05 35 4f 00 00 	mov    %rax,0x4f35(%rip)        # 408470 <proc_info_features+0x340>
  40353b:	48 8d 05 72 1c 00 00 	lea    0x1c72(%rip),%rax        # 4051b4 <_IO_stdin_used+0x1b4>
  403542:	48 89 05 17 4f 00 00 	mov    %rax,0x4f17(%rip)        # 408460 <proc_info_features+0x330>
  403549:	c7 05 2d 4f 00 00 22 	movl   $0x22,0x4f2d(%rip)        # 408480 <proc_info_features+0x350>
  403550:	00 00 00 
  403553:	48 8d 05 63 1c 00 00 	lea    0x1c63(%rip),%rax        # 4051bd <_IO_stdin_used+0x1bd>
  40355a:	48 89 05 27 4f 00 00 	mov    %rax,0x4f27(%rip)        # 408488 <proc_info_features+0x358>
  403561:	48 8d 05 5e 1c 00 00 	lea    0x1c5e(%rip),%rax        # 4051c6 <_IO_stdin_used+0x1c6>
  403568:	48 89 05 09 4f 00 00 	mov    %rax,0x4f09(%rip)        # 408478 <proc_info_features+0x348>
  40356f:	c7 05 1f 4f 00 00 23 	movl   $0x23,0x4f1f(%rip)        # 408498 <proc_info_features+0x368>
  403576:	00 00 00 
  403579:	48 8d 05 4f 1c 00 00 	lea    0x1c4f(%rip),%rax        # 4051cf <_IO_stdin_used+0x1cf>
  403580:	48 89 05 19 4f 00 00 	mov    %rax,0x4f19(%rip)        # 4084a0 <proc_info_features+0x370>
  403587:	48 8d 05 45 1c 00 00 	lea    0x1c45(%rip),%rax        # 4051d3 <_IO_stdin_used+0x1d3>
  40358e:	48 89 05 fb 4e 00 00 	mov    %rax,0x4efb(%rip)        # 408490 <proc_info_features+0x360>
  403595:	c7 05 11 4f 00 00 24 	movl   $0x24,0x4f11(%rip)        # 4084b0 <proc_info_features+0x380>
  40359c:	00 00 00 
  40359f:	48 8d 05 31 1c 00 00 	lea    0x1c31(%rip),%rax        # 4051d7 <_IO_stdin_used+0x1d7>
  4035a6:	48 89 05 0b 4f 00 00 	mov    %rax,0x4f0b(%rip)        # 4084b8 <proc_info_features+0x388>
  4035ad:	48 8d 05 27 1c 00 00 	lea    0x1c27(%rip),%rax        # 4051db <_IO_stdin_used+0x1db>
  4035b4:	48 89 05 ed 4e 00 00 	mov    %rax,0x4eed(%rip)        # 4084a8 <proc_info_features+0x378>
  4035bb:	c7 05 03 4f 00 00 25 	movl   $0x25,0x4f03(%rip)        # 4084c8 <proc_info_features+0x398>
  4035c2:	00 00 00 
  4035c5:	48 8d 05 13 1c 00 00 	lea    0x1c13(%rip),%rax        # 4051df <_IO_stdin_used+0x1df>
  4035cc:	48 89 05 fd 4e 00 00 	mov    %rax,0x4efd(%rip)        # 4084d0 <proc_info_features+0x3a0>
  4035d3:	48 8d 05 0e 1c 00 00 	lea    0x1c0e(%rip),%rax        # 4051e8 <_IO_stdin_used+0x1e8>
  4035da:	48 89 05 df 4e 00 00 	mov    %rax,0x4edf(%rip)        # 4084c0 <proc_info_features+0x390>
  4035e1:	c7 05 f5 4e 00 00 26 	movl   $0x26,0x4ef5(%rip)        # 4084e0 <proc_info_features+0x3b0>
  4035e8:	00 00 00 
  4035eb:	48 8d 05 ff 1b 00 00 	lea    0x1bff(%rip),%rax        # 4051f1 <_IO_stdin_used+0x1f1>
  4035f2:	48 89 05 ef 4e 00 00 	mov    %rax,0x4eef(%rip)        # 4084e8 <proc_info_features+0x3b8>
  4035f9:	48 8d 05 fa 1b 00 00 	lea    0x1bfa(%rip),%rax        # 4051fa <_IO_stdin_used+0x1fa>
  403600:	48 89 05 d1 4e 00 00 	mov    %rax,0x4ed1(%rip)        # 4084d8 <proc_info_features+0x3a8>
  403607:	c7 05 e7 4e 00 00 27 	movl   $0x27,0x4ee7(%rip)        # 4084f8 <proc_info_features+0x3c8>
  40360e:	00 00 00 
  403611:	48 8d 05 eb 1b 00 00 	lea    0x1beb(%rip),%rax        # 405203 <_IO_stdin_used+0x203>
  403618:	48 89 05 e1 4e 00 00 	mov    %rax,0x4ee1(%rip)        # 408500 <proc_info_features+0x3d0>
  40361f:	48 8d 05 e8 1b 00 00 	lea    0x1be8(%rip),%rax        # 40520e <_IO_stdin_used+0x20e>
  403626:	48 89 05 c3 4e 00 00 	mov    %rax,0x4ec3(%rip)        # 4084f0 <proc_info_features+0x3c0>
  40362d:	c7 05 d9 4e 00 00 28 	movl   $0x28,0x4ed9(%rip)        # 408510 <proc_info_features+0x3e0>
  403634:	00 00 00 
  403637:	48 8d 05 db 1b 00 00 	lea    0x1bdb(%rip),%rax        # 405219 <_IO_stdin_used+0x219>
  40363e:	48 89 05 d3 4e 00 00 	mov    %rax,0x4ed3(%rip)        # 408518 <proc_info_features+0x3e8>
  403645:	48 8d 05 da 1b 00 00 	lea    0x1bda(%rip),%rax        # 405226 <_IO_stdin_used+0x226>
  40364c:	48 89 05 b5 4e 00 00 	mov    %rax,0x4eb5(%rip)        # 408508 <proc_info_features+0x3d8>
  403653:	c7 05 cb 4e 00 00 29 	movl   $0x29,0x4ecb(%rip)        # 408528 <proc_info_features+0x3f8>
  40365a:	00 00 00 
  40365d:	48 8d 05 d0 1b 00 00 	lea    0x1bd0(%rip),%rax        # 405234 <_IO_stdin_used+0x234>
  403664:	48 89 05 c5 4e 00 00 	mov    %rax,0x4ec5(%rip)        # 408530 <proc_info_features+0x400>
  40366b:	48 8d 05 cf 1b 00 00 	lea    0x1bcf(%rip),%rax        # 405241 <_IO_stdin_used+0x241>
  403672:	48 89 05 a7 4e 00 00 	mov    %rax,0x4ea7(%rip)        # 408520 <proc_info_features+0x3f0>
  403679:	c7 05 bd 4e 00 00 2a 	movl   $0x2a,0x4ebd(%rip)        # 408540 <proc_info_features+0x410>
  403680:	00 00 00 
  403683:	48 8d 05 c5 1b 00 00 	lea    0x1bc5(%rip),%rax        # 40524f <_IO_stdin_used+0x24f>
  40368a:	48 89 05 b7 4e 00 00 	mov    %rax,0x4eb7(%rip)        # 408548 <proc_info_features+0x418>
  403691:	48 8d 05 c7 1b 00 00 	lea    0x1bc7(%rip),%rax        # 40525f <_IO_stdin_used+0x25f>
  403698:	48 89 05 99 4e 00 00 	mov    %rax,0x4e99(%rip)        # 408538 <proc_info_features+0x408>
  40369f:	c7 05 af 4e 00 00 2b 	movl   $0x2b,0x4eaf(%rip)        # 408558 <proc_info_features+0x428>
  4036a6:	00 00 00 
  4036a9:	48 8d 05 c0 1b 00 00 	lea    0x1bc0(%rip),%rax        # 405270 <_IO_stdin_used+0x270>
  4036b0:	48 89 05 a9 4e 00 00 	mov    %rax,0x4ea9(%rip)        # 408560 <proc_info_features+0x430>
  4036b7:	48 8d 05 bf 1b 00 00 	lea    0x1bbf(%rip),%rax        # 40527d <_IO_stdin_used+0x27d>
  4036be:	48 89 05 8b 4e 00 00 	mov    %rax,0x4e8b(%rip)        # 408550 <proc_info_features+0x420>
  4036c5:	c7 05 a1 4e 00 00 2c 	movl   $0x2c,0x4ea1(%rip)        # 408570 <proc_info_features+0x440>
  4036cc:	00 00 00 
  4036cf:	48 8d 05 b5 1b 00 00 	lea    0x1bb5(%rip),%rax        # 40528b <_IO_stdin_used+0x28b>
  4036d6:	48 89 05 9b 4e 00 00 	mov    %rax,0x4e9b(%rip)        # 408578 <proc_info_features+0x448>
  4036dd:	48 8d 05 b3 1b 00 00 	lea    0x1bb3(%rip),%rax        # 405297 <_IO_stdin_used+0x297>
  4036e4:	48 89 05 7d 4e 00 00 	mov    %rax,0x4e7d(%rip)        # 408568 <proc_info_features+0x438>
  4036eb:	c7 05 93 4e 00 00 2d 	movl   $0x2d,0x4e93(%rip)        # 408588 <proc_info_features+0x458>
  4036f2:	00 00 00 
  4036f5:	48 8d 05 a8 1b 00 00 	lea    0x1ba8(%rip),%rax        # 4052a4 <_IO_stdin_used+0x2a4>
  4036fc:	48 89 05 8d 4e 00 00 	mov    %rax,0x4e8d(%rip)        # 408590 <proc_info_features+0x460>
  403703:	48 8d 05 9f 1b 00 00 	lea    0x1b9f(%rip),%rax        # 4052a9 <_IO_stdin_used+0x2a9>
  40370a:	48 89 05 6f 4e 00 00 	mov    %rax,0x4e6f(%rip)        # 408580 <proc_info_features+0x450>
  403711:	c7 05 85 4e 00 00 2e 	movl   $0x2e,0x4e85(%rip)        # 4085a0 <proc_info_features+0x470>
  403718:	00 00 00 
  40371b:	48 8d 05 8c 1b 00 00 	lea    0x1b8c(%rip),%rax        # 4052ae <_IO_stdin_used+0x2ae>
  403722:	48 89 05 7f 4e 00 00 	mov    %rax,0x4e7f(%rip)        # 4085a8 <proc_info_features+0x478>
  403729:	48 8d 05 83 1b 00 00 	lea    0x1b83(%rip),%rax        # 4052b3 <_IO_stdin_used+0x2b3>
  403730:	48 89 05 61 4e 00 00 	mov    %rax,0x4e61(%rip)        # 408598 <proc_info_features+0x468>
  403737:	c7 05 77 4e 00 00 2f 	movl   $0x2f,0x4e77(%rip)        # 4085b8 <proc_info_features+0x488>
  40373e:	00 00 00 
  403741:	48 8d 05 70 1b 00 00 	lea    0x1b70(%rip),%rax        # 4052b8 <_IO_stdin_used+0x2b8>
  403748:	48 89 05 71 4e 00 00 	mov    %rax,0x4e71(%rip)        # 4085c0 <proc_info_features+0x490>
  40374f:	48 8d 05 6d 1b 00 00 	lea    0x1b6d(%rip),%rax        # 4052c3 <_IO_stdin_used+0x2c3>
  403756:	48 89 05 53 4e 00 00 	mov    %rax,0x4e53(%rip)        # 4085b0 <proc_info_features+0x480>
  40375d:	c7 05 69 4e 00 00 30 	movl   $0x30,0x4e69(%rip)        # 4085d0 <proc_info_features+0x4a0>
  403764:	00 00 00 
  403767:	48 8d 05 60 1b 00 00 	lea    0x1b60(%rip),%rax        # 4052ce <_IO_stdin_used+0x2ce>
  40376e:	48 89 05 63 4e 00 00 	mov    %rax,0x4e63(%rip)        # 4085d8 <proc_info_features+0x4a8>
  403775:	48 8d 05 5d 1b 00 00 	lea    0x1b5d(%rip),%rax        # 4052d9 <_IO_stdin_used+0x2d9>
  40377c:	48 89 05 45 4e 00 00 	mov    %rax,0x4e45(%rip)        # 4085c8 <proc_info_features+0x498>
  403783:	c7 05 5b 4e 00 00 31 	movl   $0x31,0x4e5b(%rip)        # 4085e8 <proc_info_features+0x4b8>
  40378a:	00 00 00 
  40378d:	48 8d 05 51 1b 00 00 	lea    0x1b51(%rip),%rax        # 4052e5 <_IO_stdin_used+0x2e5>
  403794:	48 89 05 55 4e 00 00 	mov    %rax,0x4e55(%rip)        # 4085f0 <proc_info_features+0x4c0>
  40379b:	48 8d 05 48 1b 00 00 	lea    0x1b48(%rip),%rax        # 4052ea <_IO_stdin_used+0x2ea>
  4037a2:	48 89 05 37 4e 00 00 	mov    %rax,0x4e37(%rip)        # 4085e0 <proc_info_features+0x4b0>
  4037a9:	c7 05 4d 4e 00 00 32 	movl   $0x32,0x4e4d(%rip)        # 408600 <proc_info_features+0x4d0>
  4037b0:	00 00 00 
  4037b3:	48 8d 05 35 1b 00 00 	lea    0x1b35(%rip),%rax        # 4052ef <_IO_stdin_used+0x2ef>
  4037ba:	48 89 05 47 4e 00 00 	mov    %rax,0x4e47(%rip)        # 408608 <proc_info_features+0x4d8>
  4037c1:	48 8d 05 2d 1b 00 00 	lea    0x1b2d(%rip),%rax        # 4052f5 <_IO_stdin_used+0x2f5>
  4037c8:	48 89 05 29 4e 00 00 	mov    %rax,0x4e29(%rip)        # 4085f8 <proc_info_features+0x4c8>
  4037cf:	c7 05 3f 4e 00 00 33 	movl   $0x33,0x4e3f(%rip)        # 408618 <proc_info_features+0x4e8>
  4037d6:	00 00 00 
  4037d9:	48 8d 05 1b 1b 00 00 	lea    0x1b1b(%rip),%rax        # 4052fb <_IO_stdin_used+0x2fb>
  4037e0:	48 89 05 39 4e 00 00 	mov    %rax,0x4e39(%rip)        # 408620 <proc_info_features+0x4f0>
  4037e7:	48 8d 05 11 1b 00 00 	lea    0x1b11(%rip),%rax        # 4052ff <_IO_stdin_used+0x2ff>
  4037ee:	48 89 05 1b 4e 00 00 	mov    %rax,0x4e1b(%rip)        # 408610 <proc_info_features+0x4e0>
  4037f5:	c7 05 31 4e 00 00 34 	movl   $0x34,0x4e31(%rip)        # 408630 <proc_info_features+0x500>
  4037fc:	00 00 00 
  4037ff:	48 8d 05 fd 1a 00 00 	lea    0x1afd(%rip),%rax        # 405303 <_IO_stdin_used+0x303>
  403806:	48 89 05 2b 4e 00 00 	mov    %rax,0x4e2b(%rip)        # 408638 <proc_info_features+0x508>
  40380d:	48 8d 05 f5 1a 00 00 	lea    0x1af5(%rip),%rax        # 405309 <_IO_stdin_used+0x309>
  403814:	48 89 05 0d 4e 00 00 	mov    %rax,0x4e0d(%rip)        # 408628 <proc_info_features+0x4f8>
  40381b:	c7 05 23 4e 00 00 35 	movl   $0x35,0x4e23(%rip)        # 408648 <proc_info_features+0x518>
  403822:	00 00 00 
  403825:	48 8d 05 e3 1a 00 00 	lea    0x1ae3(%rip),%rax        # 40530f <_IO_stdin_used+0x30f>
  40382c:	48 89 05 1d 4e 00 00 	mov    %rax,0x4e1d(%rip)        # 408650 <proc_info_features+0x520>
  403833:	48 8d 05 d9 1a 00 00 	lea    0x1ad9(%rip),%rax        # 405313 <_IO_stdin_used+0x313>
  40383a:	48 89 05 ff 4d 00 00 	mov    %rax,0x4dff(%rip)        # 408640 <proc_info_features+0x510>
  403841:	c7 05 15 4e 00 00 36 	movl   $0x36,0x4e15(%rip)        # 408660 <proc_info_features+0x530>
  403848:	00 00 00 
  40384b:	48 8d 05 c5 1a 00 00 	lea    0x1ac5(%rip),%rax        # 405317 <_IO_stdin_used+0x317>
  403852:	48 89 05 0f 4e 00 00 	mov    %rax,0x4e0f(%rip)        # 408668 <proc_info_features+0x538>
  403859:	48 8d 05 c0 1a 00 00 	lea    0x1ac0(%rip),%rax        # 405320 <_IO_stdin_used+0x320>
  403860:	48 89 05 f1 4d 00 00 	mov    %rax,0x4df1(%rip)        # 408658 <proc_info_features+0x528>
  403867:	c7 05 07 4e 00 00 37 	movl   $0x37,0x4e07(%rip)        # 408678 <proc_info_features+0x548>
  40386e:	00 00 00 
  403871:	48 8d 05 b1 1a 00 00 	lea    0x1ab1(%rip),%rax        # 405329 <_IO_stdin_used+0x329>
  403878:	48 89 05 01 4e 00 00 	mov    %rax,0x4e01(%rip)        # 408680 <proc_info_features+0x550>
  40387f:	48 8d 05 ab 1a 00 00 	lea    0x1aab(%rip),%rax        # 405331 <_IO_stdin_used+0x331>
  403886:	48 89 05 e3 4d 00 00 	mov    %rax,0x4de3(%rip)        # 408670 <proc_info_features+0x540>
  40388d:	c7 05 f9 4d 00 00 38 	movl   $0x38,0x4df9(%rip)        # 408690 <proc_info_features+0x560>
  403894:	00 00 00 
  403897:	48 8d 05 9b 1a 00 00 	lea    0x1a9b(%rip),%rax        # 405339 <_IO_stdin_used+0x339>
  40389e:	48 89 05 f3 4d 00 00 	mov    %rax,0x4df3(%rip)        # 408698 <proc_info_features+0x568>
  4038a5:	48 8d 05 a0 1a 00 00 	lea    0x1aa0(%rip),%rax        # 40534c <_IO_stdin_used+0x34c>
  4038ac:	48 89 05 d5 4d 00 00 	mov    %rax,0x4dd5(%rip)        # 408688 <proc_info_features+0x558>
  4038b3:	c7 05 eb 4d 00 00 3c 	movl   $0x3c,0x4deb(%rip)        # 4086a8 <proc_info_features+0x578>
  4038ba:	00 00 00 
  4038bd:	48 8d 05 9c 1a 00 00 	lea    0x1a9c(%rip),%rax        # 405360 <_IO_stdin_used+0x360>
  4038c4:	48 89 05 e5 4d 00 00 	mov    %rax,0x4de5(%rip)        # 4086b0 <proc_info_features+0x580>
  4038cb:	48 8d 05 99 1a 00 00 	lea    0x1a99(%rip),%rax        # 40536b <_IO_stdin_used+0x36b>
  4038d2:	48 89 05 c7 4d 00 00 	mov    %rax,0x4dc7(%rip)        # 4086a0 <proc_info_features+0x570>
  4038d9:	c7 05 dd 4d 00 00 40 	movl   $0x40,0x4ddd(%rip)        # 4086c0 <proc_info_features+0x590>
  4038e0:	00 00 00 
  4038e3:	48 8d 05 8d 1a 00 00 	lea    0x1a8d(%rip),%rax        # 405377 <_IO_stdin_used+0x377>
  4038ea:	48 89 05 d7 4d 00 00 	mov    %rax,0x4dd7(%rip)        # 4086c8 <proc_info_features+0x598>
  4038f1:	48 8d 05 88 1a 00 00 	lea    0x1a88(%rip),%rax        # 405380 <_IO_stdin_used+0x380>
  4038f8:	48 89 05 b9 4d 00 00 	mov    %rax,0x4db9(%rip)        # 4086b8 <proc_info_features+0x588>
  4038ff:	c7 05 cf 4d 00 00 41 	movl   $0x41,0x4dcf(%rip)        # 4086d8 <proc_info_features+0x5a8>
  403906:	00 00 00 
  403909:	48 8d 05 79 1a 00 00 	lea    0x1a79(%rip),%rax        # 405389 <_IO_stdin_used+0x389>
  403910:	48 89 05 c9 4d 00 00 	mov    %rax,0x4dc9(%rip)        # 4086e0 <proc_info_features+0x5b0>
  403917:	48 8d 05 73 1a 00 00 	lea    0x1a73(%rip),%rax        # 405391 <_IO_stdin_used+0x391>
  40391e:	48 89 05 ab 4d 00 00 	mov    %rax,0x4dab(%rip)        # 4086d0 <proc_info_features+0x5a0>
  403925:	c7 05 c1 4d 00 00 42 	movl   $0x42,0x4dc1(%rip)        # 4086f0 <proc_info_features+0x5c0>
  40392c:	00 00 00 
  40392f:	48 8d 05 63 1a 00 00 	lea    0x1a63(%rip),%rax        # 405399 <_IO_stdin_used+0x399>
  403936:	48 89 05 bb 4d 00 00 	mov    %rax,0x4dbb(%rip)        # 4086f8 <proc_info_features+0x5c8>
  40393d:	48 8d 05 5f 1a 00 00 	lea    0x1a5f(%rip),%rax        # 4053a3 <_IO_stdin_used+0x3a3>
  403944:	48 89 05 9d 4d 00 00 	mov    %rax,0x4d9d(%rip)        # 4086e8 <proc_info_features+0x5b8>
  40394b:	c7 05 b3 4d 00 00 43 	movl   $0x43,0x4db3(%rip)        # 408708 <proc_info_features+0x5d8>
  403952:	00 00 00 
  403955:	48 8d 05 51 1a 00 00 	lea    0x1a51(%rip),%rax        # 4053ad <_IO_stdin_used+0x3ad>
  40395c:	48 89 05 ad 4d 00 00 	mov    %rax,0x4dad(%rip)        # 408710 <proc_info_features+0x5e0>
  403963:	48 8d 05 4b 1a 00 00 	lea    0x1a4b(%rip),%rax        # 4053b5 <_IO_stdin_used+0x3b5>
  40396a:	48 89 05 8f 4d 00 00 	mov    %rax,0x4d8f(%rip)        # 408700 <proc_info_features+0x5d0>
  403971:	c7 05 a5 4d 00 00 44 	movl   $0x44,0x4da5(%rip)        # 408720 <proc_info_features+0x5f0>
  403978:	00 00 00 
  40397b:	48 8d 05 3b 1a 00 00 	lea    0x1a3b(%rip),%rax        # 4053bd <_IO_stdin_used+0x3bd>
  403982:	48 89 05 9f 4d 00 00 	mov    %rax,0x4d9f(%rip)        # 408728 <proc_info_features+0x5f8>
  403989:	48 8d 05 38 1a 00 00 	lea    0x1a38(%rip),%rax        # 4053c8 <_IO_stdin_used+0x3c8>
  403990:	48 89 05 81 4d 00 00 	mov    %rax,0x4d81(%rip)        # 408718 <proc_info_features+0x5e8>
  403997:	c7 05 97 4d 00 00 45 	movl   $0x45,0x4d97(%rip)        # 408738 <proc_info_features+0x608>
  40399e:	00 00 00 
  4039a1:	48 8d 05 2c 1a 00 00 	lea    0x1a2c(%rip),%rax        # 4053d4 <_IO_stdin_used+0x3d4>
  4039a8:	48 89 05 91 4d 00 00 	mov    %rax,0x4d91(%rip)        # 408740 <proc_info_features+0x610>
  4039af:	48 8d 05 25 1a 00 00 	lea    0x1a25(%rip),%rax        # 4053db <_IO_stdin_used+0x3db>
  4039b6:	48 89 05 73 4d 00 00 	mov    %rax,0x4d73(%rip)        # 408730 <proc_info_features+0x600>
  4039bd:	c7 05 89 4d 00 00 46 	movl   $0x46,0x4d89(%rip)        # 408750 <proc_info_features+0x620>
  4039c4:	00 00 00 
  4039c7:	48 8d 05 14 1a 00 00 	lea    0x1a14(%rip),%rax        # 4053e2 <_IO_stdin_used+0x3e2>
  4039ce:	48 89 05 83 4d 00 00 	mov    %rax,0x4d83(%rip)        # 408758 <proc_info_features+0x628>
  4039d5:	48 8d 05 0e 1a 00 00 	lea    0x1a0e(%rip),%rax        # 4053ea <_IO_stdin_used+0x3ea>
  4039dc:	48 89 05 65 4d 00 00 	mov    %rax,0x4d65(%rip)        # 408748 <proc_info_features+0x618>
  4039e3:	c7 05 7b 4d 00 00 47 	movl   $0x47,0x4d7b(%rip)        # 408768 <proc_info_features+0x638>
  4039ea:	00 00 00 
  4039ed:	48 8d 05 ff 19 00 00 	lea    0x19ff(%rip),%rax        # 4053f3 <_IO_stdin_used+0x3f3>
  4039f4:	48 89 05 75 4d 00 00 	mov    %rax,0x4d75(%rip)        # 408770 <proc_info_features+0x640>
  4039fb:	48 8d 05 fa 19 00 00 	lea    0x19fa(%rip),%rax        # 4053fc <_IO_stdin_used+0x3fc>
  403a02:	48 89 05 57 4d 00 00 	mov    %rax,0x4d57(%rip)        # 408760 <proc_info_features+0x630>
  403a09:	c7 05 6d 4d 00 00 48 	movl   $0x48,0x4d6d(%rip)        # 408780 <proc_info_features+0x650>
  403a10:	00 00 00 
  403a13:	48 8d 05 eb 19 00 00 	lea    0x19eb(%rip),%rax        # 405405 <_IO_stdin_used+0x405>
  403a1a:	48 89 05 67 4d 00 00 	mov    %rax,0x4d67(%rip)        # 408788 <proc_info_features+0x658>
  403a21:	48 8d 05 e6 19 00 00 	lea    0x19e6(%rip),%rax        # 40540e <_IO_stdin_used+0x40e>
  403a28:	48 89 05 49 4d 00 00 	mov    %rax,0x4d49(%rip)        # 408778 <proc_info_features+0x648>
  403a2f:	c7 05 5f 4d 00 00 49 	movl   $0x49,0x4d5f(%rip)        # 408798 <proc_info_features+0x668>
  403a36:	00 00 00 
  403a39:	48 8d 05 d7 19 00 00 	lea    0x19d7(%rip),%rax        # 405417 <_IO_stdin_used+0x417>
  403a40:	48 89 05 59 4d 00 00 	mov    %rax,0x4d59(%rip)        # 4087a0 <proc_info_features+0x670>
  403a47:	48 8d 05 d2 19 00 00 	lea    0x19d2(%rip),%rax        # 405420 <_IO_stdin_used+0x420>
  403a4e:	48 89 05 3b 4d 00 00 	mov    %rax,0x4d3b(%rip)        # 408790 <proc_info_features+0x660>
  403a55:	c7 05 51 4d 00 00 4a 	movl   $0x4a,0x4d51(%rip)        # 4087b0 <proc_info_features+0x680>
  403a5c:	00 00 00 
  403a5f:	48 8d 05 c8 19 00 00 	lea    0x19c8(%rip),%rax        # 40542e <_IO_stdin_used+0x42e>
  403a66:	48 89 05 4b 4d 00 00 	mov    %rax,0x4d4b(%rip)        # 4087b8 <proc_info_features+0x688>
  403a6d:	48 8d 05 c2 19 00 00 	lea    0x19c2(%rip),%rax        # 405436 <_IO_stdin_used+0x436>
  403a74:	48 89 05 2d 4d 00 00 	mov    %rax,0x4d2d(%rip)        # 4087a8 <proc_info_features+0x678>
  403a7b:	c7 05 43 4d 00 00 4b 	movl   $0x4b,0x4d43(%rip)        # 4087c8 <proc_info_features+0x698>
  403a82:	00 00 00 
  403a85:	48 8d 05 9d 19 00 00 	lea    0x199d(%rip),%rax        # 405429 <_IO_stdin_used+0x429>
  403a8c:	48 89 05 3d 4d 00 00 	mov    %rax,0x4d3d(%rip)        # 4087d0 <proc_info_features+0x6a0>
  403a93:	48 8d 05 97 19 00 00 	lea    0x1997(%rip),%rax        # 405431 <_IO_stdin_used+0x431>
  403a9a:	48 89 05 1f 4d 00 00 	mov    %rax,0x4d1f(%rip)        # 4087c0 <proc_info_features+0x690>
  403aa1:	c6 05 78 46 00 00 01 	movb   $0x1,0x4678(%rip)        # 408120 <__libirc_isa_info_initialized>
  403aa8:	59                   	pop    %rcx
  403aa9:	c3                   	ret
  403aaa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000403ab0 <__libirc_get_feature_bitpos>:
  403ab0:	f3 0f 1e fa          	endbr64
  403ab4:	51                   	push   %rcx
  403ab5:	89 c1                	mov    %eax,%ecx
  403ab7:	80 3d 62 46 00 00 00 	cmpb   $0x0,0x4662(%rip)        # 408120 <__libirc_isa_info_initialized>
  403abe:	75 05                	jne    403ac5 <__libirc_get_feature_bitpos+0x15>
  403ac0:	e8 2b f5 ff ff       	call   402ff0 <__libirc_isa_init_once>
  403ac5:	89 c8                	mov    %ecx,%eax
  403ac7:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  403acb:	48 8d 0d 5e 46 00 00 	lea    0x465e(%rip),%rcx        # 408130 <proc_info_features>
  403ad2:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  403ad6:	8d 41 80             	lea    -0x80(%rcx),%eax
  403ad9:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  403ade:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  403ae3:	0f 43 c1             	cmovae %ecx,%eax
  403ae6:	59                   	pop    %rcx
  403ae7:	c3                   	ret
  403ae8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  403aef:	00 

0000000000403af0 <__libirc_get_cpu_feature>:
  403af0:	f3 0f 1e fa          	endbr64
  403af4:	50                   	push   %rax
  403af5:	80 3d 24 46 00 00 00 	cmpb   $0x0,0x4624(%rip)        # 408120 <__libirc_isa_info_initialized>
  403afc:	75 05                	jne    403b03 <__libirc_get_cpu_feature+0x13>
  403afe:	e8 ed f4 ff ff       	call   402ff0 <__libirc_isa_init_once>
  403b03:	89 f0                	mov    %esi,%eax
  403b05:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  403b09:	48 8d 0d 20 46 00 00 	lea    0x4620(%rip),%rcx        # 408130 <proc_info_features>
  403b10:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  403b14:	8d 41 80             	lea    -0x80(%rcx),%eax
  403b17:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  403b1c:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  403b21:	0f 43 c1             	cmovae %ecx,%eax
  403b24:	85 c0                	test   %eax,%eax
  403b26:	78 14                	js     403b3c <__libirc_get_cpu_feature+0x4c>
  403b28:	89 c1                	mov    %eax,%ecx
  403b2a:	c1 e9 06             	shr    $0x6,%ecx
  403b2d:	48 8b 0c cf          	mov    (%rdi,%rcx,8),%rcx
  403b31:	31 d2                	xor    %edx,%edx
  403b33:	48 0f a3 c1          	bt     %rax,%rcx
  403b37:	0f 92 c2             	setb   %dl
  403b3a:	89 d0                	mov    %edx,%eax
  403b3c:	59                   	pop    %rcx
  403b3d:	c3                   	ret
  403b3e:	66 90                	xchg   %ax,%ax

0000000000403b40 <__libirc_set_cpu_feature>:
  403b40:	52                   	push   %rdx
  403b41:	56                   	push   %rsi
  403b42:	57                   	push   %rdi
  403b43:	48 89 c2             	mov    %rax,%rdx
  403b46:	80 3d d3 45 00 00 00 	cmpb   $0x0,0x45d3(%rip)        # 408120 <__libirc_isa_info_initialized>
  403b4d:	75 05                	jne    403b54 <__libirc_set_cpu_feature+0x14>
  403b4f:	e8 9c f4 ff ff       	call   402ff0 <__libirc_isa_init_once>
  403b54:	89 c8                	mov    %ecx,%eax
  403b56:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  403b5a:	48 8d 0d cf 45 00 00 	lea    0x45cf(%rip),%rcx        # 408130 <proc_info_features>
  403b61:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  403b65:	8d 41 80             	lea    -0x80(%rcx),%eax
  403b68:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  403b6d:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  403b72:	0f 43 c1             	cmovae %ecx,%eax
  403b75:	85 c0                	test   %eax,%eax
  403b77:	78 18                	js     403b91 <__libirc_set_cpu_feature+0x51>
  403b79:	89 c6                	mov    %eax,%esi
  403b7b:	c1 ee 06             	shr    $0x6,%esi
  403b7e:	83 e0 3f             	and    $0x3f,%eax
  403b81:	bf 01 00 00 00       	mov    $0x1,%edi
  403b86:	89 c1                	mov    %eax,%ecx
  403b88:	48 d3 e7             	shl    %cl,%rdi
  403b8b:	48 09 3c f2          	or     %rdi,(%rdx,%rsi,8)
  403b8f:	31 c0                	xor    %eax,%eax
  403b91:	5f                   	pop    %rdi
  403b92:	5e                   	pop    %rsi
  403b93:	5a                   	pop    %rdx
  403b94:	c3                   	ret
  403b95:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  403b9c:	00 00 00 
  403b9f:	90                   	nop

0000000000403ba0 <__libirc_handle_intel_isa_disable>:
  403ba0:	55                   	push   %rbp
  403ba1:	41 57                	push   %r15
  403ba3:	41 56                	push   %r14
  403ba5:	41 54                	push   %r12
  403ba7:	53                   	push   %rbx
  403ba8:	31 db                	xor    %ebx,%ebx
  403baa:	48 85 ff             	test   %rdi,%rdi
  403bad:	0f 84 4b 01 00 00    	je     403cfe <__libirc_handle_intel_isa_disable+0x15e>
  403bb3:	49 89 fe             	mov    %rdi,%r14
  403bb6:	48 8d 3d 9e 14 00 00 	lea    0x149e(%rip),%rdi        # 40505b <_IO_stdin_used+0x5b>
  403bbd:	e8 6e d4 ff ff       	call   401030 <getenv@plt>
  403bc2:	48 85 c0             	test   %rax,%rax
  403bc5:	0f 84 33 01 00 00    	je     403cfe <__libirc_handle_intel_isa_disable+0x15e>
  403bcb:	48 89 c2             	mov    %rax,%rdx
  403bce:	0f b6 00             	movzbl (%rax),%eax
  403bd1:	84 c0                	test   %al,%al
  403bd3:	0f 84 25 01 00 00    	je     403cfe <__libirc_handle_intel_isa_disable+0x15e>
  403bd9:	31 db                	xor    %ebx,%ebx
  403bdb:	48 8d 35 4e 45 00 00 	lea    0x454e(%rip),%rsi        # 408130 <proc_info_features>
  403be2:	31 ff                	xor    %edi,%edi
  403be4:	4c 8d 42 01          	lea    0x1(%rdx),%r8
  403be8:	41 b9 01 00 00 00    	mov    $0x1,%r9d
  403bee:	49 29 d1             	sub    %rdx,%r9
  403bf1:	49 89 d2             	mov    %rdx,%r10
  403bf4:	3c 2c                	cmp    $0x2c,%al
  403bf6:	75 1a                	jne    403c12 <__libirc_handle_intel_isa_disable+0x72>
  403bf8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  403bff:	00 
  403c00:	41 0f b6 42 01       	movzbl 0x1(%r10),%eax
  403c05:	49 ff c2             	inc    %r10
  403c08:	49 ff c0             	inc    %r8
  403c0b:	49 ff c9             	dec    %r9
  403c0e:	3c 2c                	cmp    $0x2c,%al
  403c10:	74 ee                	je     403c00 <__libirc_handle_intel_isa_disable+0x60>
  403c12:	0f b6 c0             	movzbl %al,%eax
  403c15:	85 c0                	test   %eax,%eax
  403c17:	0f 84 e1 00 00 00    	je     403cfe <__libirc_handle_intel_isa_disable+0x15e>
  403c1d:	4c 89 c2             	mov    %r8,%rdx
  403c20:	48 89 d0             	mov    %rdx,%rax
  403c23:	0f b6 0a             	movzbl (%rdx),%ecx
  403c26:	48 ff c2             	inc    %rdx
  403c29:	83 f9 2c             	cmp    $0x2c,%ecx
  403c2c:	74 12                	je     403c40 <__libirc_handle_intel_isa_disable+0xa0>
  403c2e:	85 c9                	test   %ecx,%ecx
  403c30:	74 0e                	je     403c40 <__libirc_handle_intel_isa_disable+0xa0>
  403c32:	48 89 c7             	mov    %rax,%rdi
  403c35:	eb e9                	jmp    403c20 <__libirc_handle_intel_isa_disable+0x80>
  403c37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  403c3e:	00 00 
  403c40:	49 89 fb             	mov    %rdi,%r11
  403c43:	4d 29 d3             	sub    %r10,%r11
  403c46:	48 ff ca             	dec    %rdx
  403c49:	49 ff c3             	inc    %r11
  403c4c:	75 0c                	jne    403c5a <__libirc_handle_intel_isa_disable+0xba>
  403c4e:	0f b6 02             	movzbl (%rdx),%eax
  403c51:	84 c0                	test   %al,%al
  403c53:	75 8f                	jne    403be4 <__libirc_handle_intel_isa_disable+0x44>
  403c55:	e9 a4 00 00 00       	jmp    403cfe <__libirc_handle_intel_isa_disable+0x15e>
  403c5a:	80 3d bf 44 00 00 00 	cmpb   $0x0,0x44bf(%rip)        # 408120 <__libirc_isa_info_initialized>
  403c61:	75 05                	jne    403c68 <__libirc_handle_intel_isa_disable+0xc8>
  403c63:	e8 88 f3 ff ff       	call   402ff0 <__libirc_isa_init_once>
  403c68:	4c 89 d8             	mov    %r11,%rax
  403c6b:	48 83 e0 fe          	and    $0xfffffffffffffffe,%rax
  403c6f:	49 01 f9             	add    %rdi,%r9
  403c72:	49 d1 e9             	shr    %r9
  403c75:	b9 01 00 00 00       	mov    $0x1,%ecx
  403c7a:	eb 14                	jmp    403c90 <__libirc_handle_intel_isa_disable+0xf0>
  403c7c:	0f 1f 40 00          	nopl   0x0(%rax)
  403c80:	43 80 3c 1f 00       	cmpb   $0x0,(%r15,%r11,1)
  403c85:	74 5b                	je     403ce2 <__libirc_handle_intel_isa_disable+0x142>
  403c87:	48 ff c1             	inc    %rcx
  403c8a:	48 83 f9 47          	cmp    $0x47,%rcx
  403c8e:	74 be                	je     403c4e <__libirc_handle_intel_isa_disable+0xae>
  403c90:	4c 8d 3c 49          	lea    (%rcx,%rcx,2),%r15
  403c94:	4e 8b 7c fe 10       	mov    0x10(%rsi,%r15,8),%r15
  403c99:	4d 85 ff             	test   %r15,%r15
  403c9c:	74 e9                	je     403c87 <__libirc_handle_intel_isa_disable+0xe7>
  403c9e:	49 83 fb 02          	cmp    $0x2,%r11
  403ca2:	72 2c                	jb     403cd0 <__libirc_handle_intel_isa_disable+0x130>
  403ca4:	45 31 e4             	xor    %r12d,%r12d
  403ca7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  403cae:	00 00 
  403cb0:	43 0f b6 6c 60 ff    	movzbl -0x1(%r8,%r12,2),%ebp
  403cb6:	43 3a 2c 67          	cmp    (%r15,%r12,2),%bpl
  403cba:	75 cb                	jne    403c87 <__libirc_handle_intel_isa_disable+0xe7>
  403cbc:	43 0f b6 2c 60       	movzbl (%r8,%r12,2),%ebp
  403cc1:	43 3a 6c 67 01       	cmp    0x1(%r15,%r12,2),%bpl
  403cc6:	75 bf                	jne    403c87 <__libirc_handle_intel_isa_disable+0xe7>
  403cc8:	49 ff c4             	inc    %r12
  403ccb:	4d 39 e1             	cmp    %r12,%r9
  403cce:	75 e0                	jne    403cb0 <__libirc_handle_intel_isa_disable+0x110>
  403cd0:	4c 39 d8             	cmp    %r11,%rax
  403cd3:	73 ab                	jae    403c80 <__libirc_handle_intel_isa_disable+0xe0>
  403cd5:	41 0f b6 2c 02       	movzbl (%r10,%rax,1),%ebp
  403cda:	41 3a 2c 07          	cmp    (%r15,%rax,1),%bpl
  403cde:	74 a0                	je     403c80 <__libirc_handle_intel_isa_disable+0xe0>
  403ce0:	eb a5                	jmp    403c87 <__libirc_handle_intel_isa_disable+0xe7>
  403ce2:	83 f9 02             	cmp    $0x2,%ecx
  403ce5:	0f 82 63 ff ff ff    	jb     403c4e <__libirc_handle_intel_isa_disable+0xae>
  403ceb:	4c 89 f0             	mov    %r14,%rax
  403cee:	e8 4d fe ff ff       	call   403b40 <__libirc_set_cpu_feature>
  403cf3:	83 f8 01             	cmp    $0x1,%eax
  403cf6:	83 d3 00             	adc    $0x0,%ebx
  403cf9:	e9 50 ff ff ff       	jmp    403c4e <__libirc_handle_intel_isa_disable+0xae>
  403cfe:	89 d8                	mov    %ebx,%eax
  403d00:	5b                   	pop    %rbx
  403d01:	41 5c                	pop    %r12
  403d03:	41 5e                	pop    %r14
  403d05:	41 5f                	pop    %r15
  403d07:	5d                   	pop    %rbp
  403d08:	c3                   	ret
  403d09:	0f 1f 00             	nopl   (%rax)
  403d0c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000403d10 <__libirc_get_msg>:
  403d10:	f3 0f 1e fa          	endbr64
  403d14:	53                   	push   %rbx
  403d15:	48 81 ec d0 00 00 00 	sub    $0xd0,%rsp
  403d1c:	89 f3                	mov    %esi,%ebx
  403d1e:	48 89 54 24 30       	mov    %rdx,0x30(%rsp)
  403d23:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
  403d28:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
  403d2d:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
  403d32:	84 c0                	test   %al,%al
  403d34:	74 37                	je     403d6d <__libirc_get_msg+0x5d>
  403d36:	0f 29 44 24 50       	movaps %xmm0,0x50(%rsp)
  403d3b:	0f 29 4c 24 60       	movaps %xmm1,0x60(%rsp)
  403d40:	0f 29 54 24 70       	movaps %xmm2,0x70(%rsp)
  403d45:	0f 29 9c 24 80 00 00 	movaps %xmm3,0x80(%rsp)
  403d4c:	00 
  403d4d:	0f 29 a4 24 90 00 00 	movaps %xmm4,0x90(%rsp)
  403d54:	00 
  403d55:	0f 29 ac 24 a0 00 00 	movaps %xmm5,0xa0(%rsp)
  403d5c:	00 
  403d5d:	0f 29 b4 24 b0 00 00 	movaps %xmm6,0xb0(%rsp)
  403d64:	00 
  403d65:	0f 29 bc 24 c0 00 00 	movaps %xmm7,0xc0(%rsp)
  403d6c:	00 
  403d6d:	e8 5e 00 00 00       	call   403dd0 <irc_ptr_msg>
  403d72:	85 db                	test   %ebx,%ebx
  403d74:	7e 4c                	jle    403dc2 <__libirc_get_msg+0xb2>
  403d76:	48 8d 4c 24 20       	lea    0x20(%rsp),%rcx
  403d7b:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  403d80:	48 8d 8c 24 e0 00 00 	lea    0xe0(%rsp),%rcx
  403d87:	00 
  403d88:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  403d8d:	48 b9 10 00 00 00 30 	movabs $0x3000000010,%rcx
  403d94:	00 00 00 
  403d97:	48 89 0c 24          	mov    %rcx,(%rsp)
  403d9b:	48 8d 1d 3e 4a 00 00 	lea    0x4a3e(%rip),%rbx        # 4087e0 <get_msg_buf>
  403da2:	49 89 e1             	mov    %rsp,%r9
  403da5:	be 00 02 00 00       	mov    $0x200,%esi
  403daa:	b9 00 02 00 00       	mov    $0x200,%ecx
  403daf:	48 89 df             	mov    %rbx,%rdi
  403db2:	ba 01 00 00 00       	mov    $0x1,%edx
  403db7:	49 89 c0             	mov    %rax,%r8
  403dba:	e8 61 d3 ff ff       	call   401120 <__vsnprintf_chk@plt>
  403dbf:	48 89 d8             	mov    %rbx,%rax
  403dc2:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  403dc9:	5b                   	pop    %rbx
  403dca:	c3                   	ret
  403dcb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000403dd0 <irc_ptr_msg>:
  403dd0:	41 57                	push   %r15
  403dd2:	41 56                	push   %r14
  403dd4:	41 54                	push   %r12
  403dd6:	53                   	push   %rbx
  403dd7:	48 81 ec 88 00 00 00 	sub    $0x88,%rsp
  403dde:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  403de5:	00 00 
  403de7:	48 89 84 24 80 00 00 	mov    %rax,0x80(%rsp)
  403dee:	00 
  403def:	85 ff                	test   %edi,%edi
  403df1:	74 37                	je     403e2a <irc_ptr_msg+0x5a>
  403df3:	89 fb                	mov    %edi,%ebx
  403df5:	80 3d e4 4d 00 00 00 	cmpb   $0x0,0x4de4(%rip)        # 408be0 <first_msg>
  403dfc:	74 38                	je     403e36 <irc_ptr_msg+0x66>
  403dfe:	48 63 c3             	movslq %ebx,%rax
  403e01:	48 c1 e0 04          	shl    $0x4,%rax
  403e05:	48 8d 0d e4 3b 00 00 	lea    0x3be4(%rip),%rcx        # 4079f0 <irc_msgtab>
  403e0c:	48 8b 44 08 08       	mov    0x8(%rax,%rcx,1),%rax
  403e11:	80 3d cc 4d 00 00 01 	cmpb   $0x1,0x4dcc(%rip)        # 408be4 <use_internal_msg>
  403e18:	0f 85 04 01 00 00    	jne    403f22 <irc_ptr_msg+0x152>
  403e1e:	48 8b 3d c3 4d 00 00 	mov    0x4dc3(%rip),%rdi        # 408be8 <message_catalog>
  403e25:	e9 e9 00 00 00       	jmp    403f13 <irc_ptr_msg+0x143>
  403e2a:	48 8d 05 20 12 00 00 	lea    0x1220(%rip),%rax        # 405051 <_IO_stdin_used+0x51>
  403e31:	e9 ec 00 00 00       	jmp    403f22 <irc_ptr_msg+0x152>
  403e36:	c6 05 a3 4d 00 00 01 	movb   $0x1,0x4da3(%rip)        # 408be0 <first_msg>
  403e3d:	48 8d 3d f9 15 00 00 	lea    0x15f9(%rip),%rdi        # 40543d <_IO_stdin_used+0x43d>
  403e44:	31 f6                	xor    %esi,%esi
  403e46:	e8 c5 d2 ff ff       	call   401110 <catopen@plt>
  403e4b:	48 89 c7             	mov    %rax,%rdi
  403e4e:	48 89 05 93 4d 00 00 	mov    %rax,0x4d93(%rip)        # 408be8 <message_catalog>
  403e55:	48 83 f8 ff          	cmp    $0xffffffffffffffff,%rax
  403e59:	0f 85 9a 00 00 00    	jne    403ef9 <irc_ptr_msg+0x129>
  403e5f:	48 8d 3d e3 15 00 00 	lea    0x15e3(%rip),%rdi        # 405449 <_IO_stdin_used+0x449>
  403e66:	e8 c5 d1 ff ff       	call   401030 <getenv@plt>
  403e6b:	48 85 c0             	test   %rax,%rax
  403e6e:	74 78                	je     403ee8 <irc_ptr_msg+0x118>
  403e70:	49 89 e6             	mov    %rsp,%r14
  403e73:	ba 80 00 00 00       	mov    $0x80,%edx
  403e78:	b9 80 00 00 00       	mov    $0x80,%ecx
  403e7d:	4c 89 f7             	mov    %r14,%rdi
  403e80:	48 89 c6             	mov    %rax,%rsi
  403e83:	e8 a8 d2 ff ff       	call   401130 <__strncpy_chk@plt>
  403e88:	c6 44 24 7f 00       	movb   $0x0,0x7f(%rsp)
  403e8d:	4c 89 f7             	mov    %r14,%rdi
  403e90:	be 2e 00 00 00       	mov    $0x2e,%esi
  403e95:	e8 06 d2 ff ff       	call   4010a0 <strchr@plt>
  403e9a:	48 85 c0             	test   %rax,%rax
  403e9d:	74 49                	je     403ee8 <irc_ptr_msg+0x118>
  403e9f:	49 89 c6             	mov    %rax,%r14
  403ea2:	c6 00 00             	movb   $0x0,(%rax)
  403ea5:	4c 8d 3d 9d 15 00 00 	lea    0x159d(%rip),%r15        # 405449 <_IO_stdin_used+0x449>
  403eac:	49 89 e4             	mov    %rsp,%r12
  403eaf:	4c 89 ff             	mov    %r15,%rdi
  403eb2:	4c 89 e6             	mov    %r12,%rsi
  403eb5:	ba 01 00 00 00       	mov    $0x1,%edx
  403eba:	e8 91 d1 ff ff       	call   401050 <setenv@plt>
  403ebf:	48 8d 3d 77 15 00 00 	lea    0x1577(%rip),%rdi        # 40543d <_IO_stdin_used+0x43d>
  403ec6:	31 f6                	xor    %esi,%esi
  403ec8:	e8 43 d2 ff ff       	call   401110 <catopen@plt>
  403ecd:	48 89 05 14 4d 00 00 	mov    %rax,0x4d14(%rip)        # 408be8 <message_catalog>
  403ed4:	41 c6 06 2e          	movb   $0x2e,(%r14)
  403ed8:	4c 89 ff             	mov    %r15,%rdi
  403edb:	4c 89 e6             	mov    %r12,%rsi
  403ede:	ba 01 00 00 00       	mov    $0x1,%edx
  403ee3:	e8 68 d1 ff ff       	call   401050 <setenv@plt>
  403ee8:	48 8b 3d f9 4c 00 00 	mov    0x4cf9(%rip),%rdi        # 408be8 <message_catalog>
  403eef:	48 83 ff ff          	cmp    $0xffffffffffffffff,%rdi
  403ef3:	0f 84 05 ff ff ff    	je     403dfe <irc_ptr_msg+0x2e>
  403ef9:	c6 05 e4 4c 00 00 01 	movb   $0x1,0x4ce4(%rip)        # 408be4 <use_internal_msg>
  403f00:	48 63 c3             	movslq %ebx,%rax
  403f03:	48 c1 e0 04          	shl    $0x4,%rax
  403f07:	48 8d 0d e2 3a 00 00 	lea    0x3ae2(%rip),%rcx        # 4079f0 <irc_msgtab>
  403f0e:	48 8b 44 08 08       	mov    0x8(%rax,%rcx,1),%rax
  403f13:	be 01 00 00 00       	mov    $0x1,%esi
  403f18:	89 da                	mov    %ebx,%edx
  403f1a:	48 89 c1             	mov    %rax,%rcx
  403f1d:	e8 4e d2 ff ff       	call   401170 <catgets@plt>
  403f22:	64 48 8b 0c 25 28 00 	mov    %fs:0x28,%rcx
  403f29:	00 00 
  403f2b:	48 3b 8c 24 80 00 00 	cmp    0x80(%rsp),%rcx
  403f32:	00 
  403f33:	75 0f                	jne    403f44 <irc_ptr_msg+0x174>
  403f35:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
  403f3c:	5b                   	pop    %rbx
  403f3d:	41 5c                	pop    %r12
  403f3f:	41 5e                	pop    %r14
  403f41:	41 5f                	pop    %r15
  403f43:	c3                   	ret
  403f44:	e8 47 d1 ff ff       	call   401090 <__stack_chk_fail@plt>
  403f49:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000403f50 <__libirc_print>:
  403f50:	f3 0f 1e fa          	endbr64
  403f54:	55                   	push   %rbp
  403f55:	41 56                	push   %r14
  403f57:	53                   	push   %rbx
  403f58:	48 81 ec d0 00 00 00 	sub    $0xd0,%rsp
  403f5f:	89 fb                	mov    %edi,%ebx
  403f61:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
  403f66:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
  403f6b:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
  403f70:	84 c0                	test   %al,%al
  403f72:	74 37                	je     403fab <__libirc_print+0x5b>
  403f74:	0f 29 44 24 50       	movaps %xmm0,0x50(%rsp)
  403f79:	0f 29 4c 24 60       	movaps %xmm1,0x60(%rsp)
  403f7e:	0f 29 54 24 70       	movaps %xmm2,0x70(%rsp)
  403f83:	0f 29 9c 24 80 00 00 	movaps %xmm3,0x80(%rsp)
  403f8a:	00 
  403f8b:	0f 29 a4 24 90 00 00 	movaps %xmm4,0x90(%rsp)
  403f92:	00 
  403f93:	0f 29 ac 24 a0 00 00 	movaps %xmm5,0xa0(%rsp)
  403f9a:	00 
  403f9b:	0f 29 b4 24 b0 00 00 	movaps %xmm6,0xb0(%rsp)
  403fa2:	00 
  403fa3:	0f 29 bc 24 c0 00 00 	movaps %xmm7,0xc0(%rsp)
  403faa:	00 
  403fab:	85 f6                	test   %esi,%esi
  403fad:	0f 84 80 00 00 00    	je     404033 <__libirc_print+0xe3>
  403fb3:	89 d5                	mov    %edx,%ebp
  403fb5:	89 f7                	mov    %esi,%edi
  403fb7:	e8 14 fe ff ff       	call   403dd0 <irc_ptr_msg>
  403fbc:	85 ed                	test   %ebp,%ebp
  403fbe:	7e 4c                	jle    40400c <__libirc_print+0xbc>
  403fc0:	48 8d 4c 24 20       	lea    0x20(%rsp),%rcx
  403fc5:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  403fca:	48 8d 8c 24 f0 00 00 	lea    0xf0(%rsp),%rcx
  403fd1:	00 
  403fd2:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  403fd7:	48 b9 18 00 00 00 30 	movabs $0x3000000018,%rcx
  403fde:	00 00 00 
  403fe1:	48 89 0c 24          	mov    %rcx,(%rsp)
  403fe5:	4c 8d 35 f4 49 00 00 	lea    0x49f4(%rip),%r14        # 4089e0 <print_buf>
  403fec:	49 89 e1             	mov    %rsp,%r9
  403fef:	be 00 02 00 00       	mov    $0x200,%esi
  403ff4:	b9 00 02 00 00       	mov    $0x200,%ecx
  403ff9:	4c 89 f7             	mov    %r14,%rdi
  403ffc:	ba 01 00 00 00       	mov    $0x1,%edx
  404001:	49 89 c0             	mov    %rax,%r8
  404004:	e8 17 d1 ff ff       	call   401120 <__vsnprintf_chk@plt>
  404009:	4c 89 f0             	mov    %r14,%rax
  40400c:	83 fb 01             	cmp    $0x1,%ebx
  40400f:	75 4f                	jne    404060 <__libirc_print+0x110>
  404011:	48 8b 0d c0 3f 00 00 	mov    0x3fc0(%rip),%rcx        # 407fd8 <stderr@GLIBC_2.2.5-0x108>
  404018:	48 8b 39             	mov    (%rcx),%rdi
  40401b:	48 8d 15 17 14 00 00 	lea    0x1417(%rip),%rdx        # 405439 <_IO_stdin_used+0x439>
  404022:	be 01 00 00 00       	mov    $0x1,%esi
  404027:	48 89 c1             	mov    %rax,%rcx
  40402a:	31 c0                	xor    %eax,%eax
  40402c:	e8 5f d1 ff ff       	call   401190 <__fprintf_chk@plt>
  404031:	eb 43                	jmp    404076 <__libirc_print+0x126>
  404033:	83 fb 01             	cmp    $0x1,%ebx
  404036:	75 4a                	jne    404082 <__libirc_print+0x132>
  404038:	48 8b 05 99 3f 00 00 	mov    0x3f99(%rip),%rax        # 407fd8 <stderr@GLIBC_2.2.5-0x108>
  40403f:	48 8b 38             	mov    (%rax),%rdi
  404042:	48 8d 15 07 10 00 00 	lea    0x1007(%rip),%rdx        # 405050 <_IO_stdin_used+0x50>
  404049:	be 01 00 00 00       	mov    $0x1,%esi
  40404e:	31 c0                	xor    %eax,%eax
  404050:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  404057:	5b                   	pop    %rbx
  404058:	41 5e                	pop    %r14
  40405a:	5d                   	pop    %rbp
  40405b:	e9 30 d1 ff ff       	jmp    401190 <__fprintf_chk@plt>
  404060:	48 8d 35 d2 13 00 00 	lea    0x13d2(%rip),%rsi        # 405439 <_IO_stdin_used+0x439>
  404067:	bf 01 00 00 00       	mov    $0x1,%edi
  40406c:	48 89 c2             	mov    %rax,%rdx
  40406f:	31 c0                	xor    %eax,%eax
  404071:	e8 ca d0 ff ff       	call   401140 <__printf_chk@plt>
  404076:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  40407d:	5b                   	pop    %rbx
  40407e:	41 5e                	pop    %r14
  404080:	5d                   	pop    %rbp
  404081:	c3                   	ret
  404082:	48 8d 35 c7 0f 00 00 	lea    0xfc7(%rip),%rsi        # 405050 <_IO_stdin_used+0x50>
  404089:	bf 01 00 00 00       	mov    $0x1,%edi
  40408e:	31 c0                	xor    %eax,%eax
  404090:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  404097:	5b                   	pop    %rbx
  404098:	41 5e                	pop    %r14
  40409a:	5d                   	pop    %rbp
  40409b:	e9 a0 d0 ff ff       	jmp    401140 <__printf_chk@plt>

Disassembly of section .fini:

00000000004040a0 <_fini>:
  4040a0:	48 83 ec 08          	sub    $0x8,%rsp
  4040a4:	48 83 c4 08          	add    $0x8,%rsp
  4040a8:	c3                   	ret
