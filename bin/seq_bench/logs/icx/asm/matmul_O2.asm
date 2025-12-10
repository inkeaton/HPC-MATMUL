
bin/seq_bench/icx/matmul_O2:     file format elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
  401000:	48 83 ec 08          	sub    $0x8,%rsp
  401004:	48 8b 05 c5 2f 00 00 	mov    0x2fc5(%rip),%rax        # 403fd0 <__gmon_start__@Base>
  40100b:	48 85 c0             	test   %rax,%rax
  40100e:	74 02                	je     401012 <_init+0x12>
  401010:	ff d0                	call   *%rax
  401012:	48 83 c4 08          	add    $0x8,%rsp
  401016:	c3                   	ret

Disassembly of section .plt:

0000000000401020 <free@plt-0x10>:
  401020:	ff 35 ca 2f 00 00    	push   0x2fca(%rip)        # 403ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
  401026:	ff 25 cc 2f 00 00    	jmp    *0x2fcc(%rip)        # 403ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
  40102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401030 <free@plt>:
  401030:	ff 25 ca 2f 00 00    	jmp    *0x2fca(%rip)        # 404000 <free@GLIBC_2.2.5>
  401036:	68 00 00 00 00       	push   $0x0
  40103b:	e9 e0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401040 <clock@plt>:
  401040:	ff 25 c2 2f 00 00    	jmp    *0x2fc2(%rip)        # 404008 <clock@GLIBC_2.2.5>
  401046:	68 01 00 00 00       	push   $0x1
  40104b:	e9 d0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401050 <fclose@plt>:
  401050:	ff 25 ba 2f 00 00    	jmp    *0x2fba(%rip)        # 404010 <fclose@GLIBC_2.2.5>
  401056:	68 02 00 00 00       	push   $0x2
  40105b:	e9 c0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401060 <fputc@plt>:
  401060:	ff 25 b2 2f 00 00    	jmp    *0x2fb2(%rip)        # 404018 <fputc@GLIBC_2.2.5>
  401066:	68 03 00 00 00       	push   $0x3
  40106b:	e9 b0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401070 <calloc@plt>:
  401070:	ff 25 aa 2f 00 00    	jmp    *0x2faa(%rip)        # 404020 <calloc@GLIBC_2.2.5>
  401076:	68 04 00 00 00       	push   $0x4
  40107b:	e9 a0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401080 <fprintf@plt>:
  401080:	ff 25 a2 2f 00 00    	jmp    *0x2fa2(%rip)        # 404028 <fprintf@GLIBC_2.2.5>
  401086:	68 05 00 00 00       	push   $0x5
  40108b:	e9 90 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401090 <malloc@plt>:
  401090:	ff 25 9a 2f 00 00    	jmp    *0x2f9a(%rip)        # 404030 <malloc@GLIBC_2.2.5>
  401096:	68 06 00 00 00       	push   $0x6
  40109b:	e9 80 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010a0 <fopen@plt>:
  4010a0:	ff 25 92 2f 00 00    	jmp    *0x2f92(%rip)        # 404038 <fopen@GLIBC_2.2.5>
  4010a6:	68 07 00 00 00       	push   $0x7
  4010ab:	e9 70 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010b0 <perror@plt>:
  4010b0:	ff 25 8a 2f 00 00    	jmp    *0x2f8a(%rip)        # 404040 <perror@GLIBC_2.2.5>
  4010b6:	68 08 00 00 00       	push   $0x8
  4010bb:	e9 60 ff ff ff       	jmp    401020 <_init+0x20>

Disassembly of section .plt.got:

00000000004010c0 <__cxa_finalize@plt>:
  4010c0:	ff 25 1a 2f 00 00    	jmp    *0x2f1a(%rip)        # 403fe0 <__cxa_finalize@GLIBC_2.2.5>
  4010c6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004010d0 <_start>:
  4010d0:	31 ed                	xor    %ebp,%ebp
  4010d2:	49 89 d1             	mov    %rdx,%r9
  4010d5:	5e                   	pop    %rsi
  4010d6:	48 89 e2             	mov    %rsp,%rdx
  4010d9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4010dd:	50                   	push   %rax
  4010de:	54                   	push   %rsp
  4010df:	45 31 c0             	xor    %r8d,%r8d
  4010e2:	31 c9                	xor    %ecx,%ecx
  4010e4:	48 c7 c7 c0 11 40 00 	mov    $0x4011c0,%rdi
  4010eb:	ff 15 cf 2e 00 00    	call   *0x2ecf(%rip)        # 403fc0 <__libc_start_main@GLIBC_2.34>
  4010f1:	f4                   	hlt
  4010f2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4010f9:	00 00 00 
  4010fc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401100 <deregister_tm_clones>:
  401100:	48 8d 3d 51 2f 00 00 	lea    0x2f51(%rip),%rdi        # 404058 <__TMC_END__>
  401107:	48 8d 05 4a 2f 00 00 	lea    0x2f4a(%rip),%rax        # 404058 <__TMC_END__>
  40110e:	48 39 f8             	cmp    %rdi,%rax
  401111:	74 15                	je     401128 <deregister_tm_clones+0x28>
  401113:	48 8b 05 ae 2e 00 00 	mov    0x2eae(%rip),%rax        # 403fc8 <_ITM_deregisterTMCloneTable@Base>
  40111a:	48 85 c0             	test   %rax,%rax
  40111d:	74 09                	je     401128 <deregister_tm_clones+0x28>
  40111f:	ff e0                	jmp    *%rax
  401121:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401128:	c3                   	ret
  401129:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401130 <register_tm_clones>:
  401130:	48 8d 3d 21 2f 00 00 	lea    0x2f21(%rip),%rdi        # 404058 <__TMC_END__>
  401137:	48 8d 35 1a 2f 00 00 	lea    0x2f1a(%rip),%rsi        # 404058 <__TMC_END__>
  40113e:	48 29 fe             	sub    %rdi,%rsi
  401141:	48 89 f0             	mov    %rsi,%rax
  401144:	48 c1 ee 3f          	shr    $0x3f,%rsi
  401148:	48 c1 f8 03          	sar    $0x3,%rax
  40114c:	48 01 c6             	add    %rax,%rsi
  40114f:	48 d1 fe             	sar    %rsi
  401152:	74 14                	je     401168 <register_tm_clones+0x38>
  401154:	48 8b 05 7d 2e 00 00 	mov    0x2e7d(%rip),%rax        # 403fd8 <_ITM_registerTMCloneTable@Base>
  40115b:	48 85 c0             	test   %rax,%rax
  40115e:	74 08                	je     401168 <register_tm_clones+0x38>
  401160:	ff e0                	jmp    *%rax
  401162:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401168:	c3                   	ret
  401169:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401170 <__do_global_dtors_aux>:
  401170:	f3 0f 1e fa          	endbr64
  401174:	80 3d ed 2e 00 00 00 	cmpb   $0x0,0x2eed(%rip)        # 404068 <completed.0>
  40117b:	75 2b                	jne    4011a8 <__do_global_dtors_aux+0x38>
  40117d:	55                   	push   %rbp
  40117e:	48 83 3d 5a 2e 00 00 	cmpq   $0x0,0x2e5a(%rip)        # 403fe0 <__cxa_finalize@GLIBC_2.2.5>
  401185:	00 
  401186:	48 89 e5             	mov    %rsp,%rbp
  401189:	74 0c                	je     401197 <__do_global_dtors_aux+0x27>
  40118b:	48 8b 3d be 2e 00 00 	mov    0x2ebe(%rip),%rdi        # 404050 <__dso_handle>
  401192:	e8 29 ff ff ff       	call   4010c0 <__cxa_finalize@plt>
  401197:	e8 64 ff ff ff       	call   401100 <deregister_tm_clones>
  40119c:	c6 05 c5 2e 00 00 01 	movb   $0x1,0x2ec5(%rip)        # 404068 <completed.0>
  4011a3:	5d                   	pop    %rbp
  4011a4:	c3                   	ret
  4011a5:	0f 1f 00             	nopl   (%rax)
  4011a8:	c3                   	ret
  4011a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004011b0 <frame_dummy>:
  4011b0:	f3 0f 1e fa          	endbr64
  4011b4:	e9 77 ff ff ff       	jmp    401130 <register_tm_clones>
  4011b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004011c0 <main>:
  4011c0:	55                   	push   %rbp
  4011c1:	41 57                	push   %r15
  4011c3:	41 56                	push   %r14
  4011c5:	41 55                	push   %r13
  4011c7:	41 54                	push   %r12
  4011c9:	53                   	push   %rbx
  4011ca:	48 83 ec 18          	sub    $0x18,%rsp
  4011ce:	0f ae 5c 24 0c       	stmxcsr 0xc(%rsp)
  4011d3:	81 4c 24 0c 40 80 00 	orl    $0x8040,0xc(%rsp)
  4011da:	00 
  4011db:	0f ae 54 24 0c       	ldmxcsr 0xc(%rsp)
  4011e0:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4011e5:	e8 a6 fe ff ff       	call   401090 <malloc@plt>
  4011ea:	48 89 c3             	mov    %rax,%rbx
  4011ed:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4011f2:	e8 99 fe ff ff       	call   401090 <malloc@plt>
  4011f7:	49 89 c4             	mov    %rax,%r12
  4011fa:	49 89 c6             	mov    %rax,%r14
  4011fd:	bf 01 00 00 00       	mov    $0x1,%edi
  401202:	be 00 c2 eb 0b       	mov    $0xbebc200,%esi
  401207:	e8 64 fe ff ff       	call   401070 <calloc@plt>
  40120c:	49 89 c7             	mov    %rax,%r15
  40120f:	b8 30 00 00 00       	mov    $0x30,%eax
  401214:	66 0f 28 05 f4 0d 00 	movapd 0xdf4(%rip),%xmm0        # 402010 <_IO_stdin_used+0x10>
  40121b:	00 
  40121c:	66 0f 28 0d fc 0d 00 	movapd 0xdfc(%rip),%xmm1        # 402020 <_IO_stdin_used+0x20>
  401223:	00 
  401224:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
  40122b:	00 00 00 00 00 
  401230:	66 0f 11 44 03 d0    	movupd %xmm0,-0x30(%rbx,%rax,1)
  401236:	66 41 0f 11 4c 06 d0 	movupd %xmm1,-0x30(%r14,%rax,1)
  40123d:	66 0f 11 44 03 e0    	movupd %xmm0,-0x20(%rbx,%rax,1)
  401243:	66 41 0f 11 4c 06 e0 	movupd %xmm1,-0x20(%r14,%rax,1)
  40124a:	66 0f 11 44 03 f0    	movupd %xmm0,-0x10(%rbx,%rax,1)
  401250:	66 41 0f 11 4c 06 f0 	movupd %xmm1,-0x10(%r14,%rax,1)
  401257:	66 0f 11 04 03       	movupd %xmm0,(%rbx,%rax,1)
  40125c:	66 41 0f 11 0c 06    	movupd %xmm1,(%r14,%rax,1)
  401262:	48 83 c0 40          	add    $0x40,%rax
  401266:	48 3d 30 c2 eb 0b    	cmp    $0xbebc230,%rax
  40126c:	75 c2                	jne    401230 <main+0x70>
  40126e:	e8 cd fd ff ff       	call   401040 <clock@plt>
  401273:	49 89 c5             	mov    %rax,%r13
  401276:	49 83 c4 10          	add    $0x10,%r12
  40127a:	49 8d 47 10          	lea    0x10(%r15),%rax
  40127e:	31 c9                	xor    %ecx,%ecx
  401280:	4c 89 e2             	mov    %r12,%rdx
  401283:	31 f6                	xor    %esi,%esi
  401285:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  40128c:	00 00 00 00 
  401290:	48 69 f9 40 9c 00 00 	imul   $0x9c40,%rcx,%rdi
  401297:	48 01 df             	add    %rbx,%rdi
  40129a:	f2 0f 10 04 f7       	movsd  (%rdi,%rsi,8),%xmm0
  40129f:	66 0f 14 c0          	unpcklpd %xmm0,%xmm0
  4012a3:	48 c7 c7 fe ff ff ff 	mov    $0xfffffffffffffffe,%rdi
  4012aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4012b0:	66 0f 10 0c fa       	movupd (%rdx,%rdi,8),%xmm1
  4012b5:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  4012b9:	66 0f 10 14 f8       	movupd (%rax,%rdi,8),%xmm2
  4012be:	66 0f 10 5c f8 10    	movupd 0x10(%rax,%rdi,8),%xmm3
  4012c4:	66 0f 58 d1          	addpd  %xmm1,%xmm2
  4012c8:	66 0f 11 14 f8       	movupd %xmm2,(%rax,%rdi,8)
  4012cd:	66 0f 10 4c fa 10    	movupd 0x10(%rdx,%rdi,8),%xmm1
  4012d3:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  4012d7:	66 0f 58 cb          	addpd  %xmm3,%xmm1
  4012db:	66 0f 11 4c f8 10    	movupd %xmm1,0x10(%rax,%rdi,8)
  4012e1:	48 83 c7 04          	add    $0x4,%rdi
  4012e5:	48 81 ff 86 13 00 00 	cmp    $0x1386,%rdi
  4012ec:	72 c2                	jb     4012b0 <main+0xf0>
  4012ee:	48 8d 7e 01          	lea    0x1(%rsi),%rdi
  4012f2:	48 81 c2 40 9c 00 00 	add    $0x9c40,%rdx
  4012f9:	48 81 fe 87 13 00 00 	cmp    $0x1387,%rsi
  401300:	48 89 fe             	mov    %rdi,%rsi
  401303:	75 8b                	jne    401290 <main+0xd0>
  401305:	48 8d 51 01          	lea    0x1(%rcx),%rdx
  401309:	48 05 40 9c 00 00    	add    $0x9c40,%rax
  40130f:	48 81 f9 87 13 00 00 	cmp    $0x1387,%rcx
  401316:	48 89 d1             	mov    %rdx,%rcx
  401319:	0f 85 61 ff ff ff    	jne    401280 <main+0xc0>
  40131f:	e8 1c fd ff ff       	call   401040 <clock@plt>
  401324:	4c 29 e8             	sub    %r13,%rax
  401327:	0f 57 c0             	xorps  %xmm0,%xmm0
  40132a:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
  40132f:	f2 0f 59 05 f9 0c 00 	mulsd  0xcf9(%rip),%xmm0        # 402030 <_IO_stdin_used+0x30>
  401336:	00 
  401337:	48 8b 3d 22 2d 00 00 	mov    0x2d22(%rip),%rdi        # 404060 <stderr@GLIBC_2.2.5>
  40133e:	be 38 20 40 00       	mov    $0x402038,%esi
  401343:	ba 88 13 00 00       	mov    $0x1388,%edx
  401348:	b0 01                	mov    $0x1,%al
  40134a:	e8 31 fd ff ff       	call   401080 <fprintf@plt>
  40134f:	bf 53 20 40 00       	mov    $0x402053,%edi
  401354:	be 5f 20 40 00       	mov    $0x40205f,%esi
  401359:	e8 42 fd ff ff       	call   4010a0 <fopen@plt>
  40135e:	48 85 c0             	test   %rax,%rax
  401361:	0f 84 55 01 00 00    	je     4014bc <main+0x2fc>
  401367:	49 89 c4             	mov    %rax,%r12
  40136a:	45 31 ed             	xor    %r13d,%r13d
  40136d:	be 67 20 40 00       	mov    $0x402067,%esi
  401372:	48 89 c7             	mov    %rax,%rdi
  401375:	ba 88 13 00 00       	mov    $0x1388,%edx
  40137a:	31 c0                	xor    %eax,%eax
  40137c:	e8 ff fc ff ff       	call   401080 <fprintf@plt>
  401381:	4c 89 7c 24 10       	mov    %r15,0x10(%rsp)
  401386:	4c 89 fd             	mov    %r15,%rbp
  401389:	48 83 c5 38          	add    $0x38,%rbp
  40138d:	0f 1f 00             	nopl   (%rax)
  401390:	45 31 ff             	xor    %r15d,%r15d
  401393:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  40139a:	84 00 00 00 00 00 
  4013a0:	f2 42 0f 10 44 3d c8 	movsd  -0x38(%rbp,%r15,1),%xmm0
  4013a7:	be 6c 20 40 00       	mov    $0x40206c,%esi
  4013ac:	4c 89 e7             	mov    %r12,%rdi
  4013af:	b0 01                	mov    $0x1,%al
  4013b1:	e8 ca fc ff ff       	call   401080 <fprintf@plt>
  4013b6:	f2 42 0f 10 44 3d d0 	movsd  -0x30(%rbp,%r15,1),%xmm0
  4013bd:	be 6c 20 40 00       	mov    $0x40206c,%esi
  4013c2:	4c 89 e7             	mov    %r12,%rdi
  4013c5:	b0 01                	mov    $0x1,%al
  4013c7:	e8 b4 fc ff ff       	call   401080 <fprintf@plt>
  4013cc:	f2 42 0f 10 44 3d d8 	movsd  -0x28(%rbp,%r15,1),%xmm0
  4013d3:	be 6c 20 40 00       	mov    $0x40206c,%esi
  4013d8:	4c 89 e7             	mov    %r12,%rdi
  4013db:	b0 01                	mov    $0x1,%al
  4013dd:	e8 9e fc ff ff       	call   401080 <fprintf@plt>
  4013e2:	f2 42 0f 10 44 3d e0 	movsd  -0x20(%rbp,%r15,1),%xmm0
  4013e9:	be 6c 20 40 00       	mov    $0x40206c,%esi
  4013ee:	4c 89 e7             	mov    %r12,%rdi
  4013f1:	b0 01                	mov    $0x1,%al
  4013f3:	e8 88 fc ff ff       	call   401080 <fprintf@plt>
  4013f8:	f2 42 0f 10 44 3d e8 	movsd  -0x18(%rbp,%r15,1),%xmm0
  4013ff:	be 6c 20 40 00       	mov    $0x40206c,%esi
  401404:	4c 89 e7             	mov    %r12,%rdi
  401407:	b0 01                	mov    $0x1,%al
  401409:	e8 72 fc ff ff       	call   401080 <fprintf@plt>
  40140e:	f2 42 0f 10 44 3d f0 	movsd  -0x10(%rbp,%r15,1),%xmm0
  401415:	be 6c 20 40 00       	mov    $0x40206c,%esi
  40141a:	4c 89 e7             	mov    %r12,%rdi
  40141d:	b0 01                	mov    $0x1,%al
  40141f:	e8 5c fc ff ff       	call   401080 <fprintf@plt>
  401424:	f2 42 0f 10 44 3d f8 	movsd  -0x8(%rbp,%r15,1),%xmm0
  40142b:	be 6c 20 40 00       	mov    $0x40206c,%esi
  401430:	4c 89 e7             	mov    %r12,%rdi
  401433:	b0 01                	mov    $0x1,%al
  401435:	e8 46 fc ff ff       	call   401080 <fprintf@plt>
  40143a:	f2 42 0f 10 44 3d 00 	movsd  0x0(%rbp,%r15,1),%xmm0
  401441:	be 6c 20 40 00       	mov    $0x40206c,%esi
  401446:	4c 89 e7             	mov    %r12,%rdi
  401449:	b0 01                	mov    $0x1,%al
  40144b:	e8 30 fc ff ff       	call   401080 <fprintf@plt>
  401450:	49 83 c7 40          	add    $0x40,%r15
  401454:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  40145b:	0f 85 3f ff ff ff    	jne    4013a0 <main+0x1e0>
  401461:	bf 0a 00 00 00       	mov    $0xa,%edi
  401466:	4c 89 e6             	mov    %r12,%rsi
  401469:	e8 f2 fb ff ff       	call   401060 <fputc@plt>
  40146e:	49 8d 45 01          	lea    0x1(%r13),%rax
  401472:	48 81 c5 40 9c 00 00 	add    $0x9c40,%rbp
  401479:	49 81 fd e7 03 00 00 	cmp    $0x3e7,%r13
  401480:	49 89 c5             	mov    %rax,%r13
  401483:	0f 85 07 ff ff ff    	jne    401390 <main+0x1d0>
  401489:	4c 89 e7             	mov    %r12,%rdi
  40148c:	e8 bf fb ff ff       	call   401050 <fclose@plt>
  401491:	48 89 df             	mov    %rbx,%rdi
  401494:	e8 97 fb ff ff       	call   401030 <free@plt>
  401499:	4c 89 f7             	mov    %r14,%rdi
  40149c:	e8 8f fb ff ff       	call   401030 <free@plt>
  4014a1:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4014a6:	e8 85 fb ff ff       	call   401030 <free@plt>
  4014ab:	31 c0                	xor    %eax,%eax
  4014ad:	48 83 c4 18          	add    $0x18,%rsp
  4014b1:	5b                   	pop    %rbx
  4014b2:	41 5c                	pop    %r12
  4014b4:	41 5d                	pop    %r13
  4014b6:	41 5e                	pop    %r14
  4014b8:	41 5f                	pop    %r15
  4014ba:	5d                   	pop    %rbp
  4014bb:	c3                   	ret
  4014bc:	bf 61 20 40 00       	mov    $0x402061,%edi
  4014c1:	e8 ea fb ff ff       	call   4010b0 <perror@plt>
  4014c6:	b8 01 00 00 00       	mov    $0x1,%eax
  4014cb:	eb e0                	jmp    4014ad <main+0x2ed>

Disassembly of section .fini:

00000000004014d0 <_fini>:
  4014d0:	48 83 ec 08          	sub    $0x8,%rsp
  4014d4:	48 83 c4 08          	add    $0x8,%rsp
  4014d8:	c3                   	ret
