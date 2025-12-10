
bin/seq_bench/icc/matmul_O0:     file format elf64-x86-64


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

0000000000401060 <fprintf@plt>:
  401060:	ff 25 b2 2f 00 00    	jmp    *0x2fb2(%rip)        # 404018 <fprintf@GLIBC_2.2.5>
  401066:	68 03 00 00 00       	push   $0x3
  40106b:	e9 b0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401070 <malloc@plt>:
  401070:	ff 25 aa 2f 00 00    	jmp    *0x2faa(%rip)        # 404020 <malloc@GLIBC_2.2.5>
  401076:	68 04 00 00 00       	push   $0x4
  40107b:	e9 a0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401080 <fopen@plt>:
  401080:	ff 25 a2 2f 00 00    	jmp    *0x2fa2(%rip)        # 404028 <fopen@GLIBC_2.2.5>
  401086:	68 05 00 00 00       	push   $0x5
  40108b:	e9 90 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401090 <perror@plt>:
  401090:	ff 25 9a 2f 00 00    	jmp    *0x2f9a(%rip)        # 404030 <perror@GLIBC_2.2.5>
  401096:	68 06 00 00 00       	push   $0x6
  40109b:	e9 80 ff ff ff       	jmp    401020 <_init+0x20>

Disassembly of section .plt.got:

00000000004010a0 <__cxa_finalize@plt>:
  4010a0:	ff 25 3a 2f 00 00    	jmp    *0x2f3a(%rip)        # 403fe0 <__cxa_finalize@GLIBC_2.2.5>
  4010a6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004010b0 <_start>:
  4010b0:	31 ed                	xor    %ebp,%ebp
  4010b2:	49 89 d1             	mov    %rdx,%r9
  4010b5:	5e                   	pop    %rsi
  4010b6:	48 89 e2             	mov    %rsp,%rdx
  4010b9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4010bd:	50                   	push   %rax
  4010be:	54                   	push   %rsp
  4010bf:	45 31 c0             	xor    %r8d,%r8d
  4010c2:	31 c9                	xor    %ecx,%ecx
  4010c4:	48 c7 c7 9c 11 40 00 	mov    $0x40119c,%rdi
  4010cb:	ff 15 ef 2e 00 00    	call   *0x2eef(%rip)        # 403fc0 <__libc_start_main@GLIBC_2.34>
  4010d1:	f4                   	hlt
  4010d2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4010d9:	00 00 00 
  4010dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004010e0 <deregister_tm_clones>:
  4010e0:	48 8d 3d 61 2f 00 00 	lea    0x2f61(%rip),%rdi        # 404048 <__TMC_END__>
  4010e7:	48 8d 05 5a 2f 00 00 	lea    0x2f5a(%rip),%rax        # 404048 <__TMC_END__>
  4010ee:	48 39 f8             	cmp    %rdi,%rax
  4010f1:	74 15                	je     401108 <deregister_tm_clones+0x28>
  4010f3:	48 8b 05 ce 2e 00 00 	mov    0x2ece(%rip),%rax        # 403fc8 <_ITM_deregisterTMCloneTable@Base>
  4010fa:	48 85 c0             	test   %rax,%rax
  4010fd:	74 09                	je     401108 <deregister_tm_clones+0x28>
  4010ff:	ff e0                	jmp    *%rax
  401101:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401108:	c3                   	ret
  401109:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401110 <register_tm_clones>:
  401110:	48 8d 3d 31 2f 00 00 	lea    0x2f31(%rip),%rdi        # 404048 <__TMC_END__>
  401117:	48 8d 35 2a 2f 00 00 	lea    0x2f2a(%rip),%rsi        # 404048 <__TMC_END__>
  40111e:	48 29 fe             	sub    %rdi,%rsi
  401121:	48 89 f0             	mov    %rsi,%rax
  401124:	48 c1 ee 3f          	shr    $0x3f,%rsi
  401128:	48 c1 f8 03          	sar    $0x3,%rax
  40112c:	48 01 c6             	add    %rax,%rsi
  40112f:	48 d1 fe             	sar    %rsi
  401132:	74 14                	je     401148 <register_tm_clones+0x38>
  401134:	48 8b 05 9d 2e 00 00 	mov    0x2e9d(%rip),%rax        # 403fd8 <_ITM_registerTMCloneTable@Base>
  40113b:	48 85 c0             	test   %rax,%rax
  40113e:	74 08                	je     401148 <register_tm_clones+0x38>
  401140:	ff e0                	jmp    *%rax
  401142:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401148:	c3                   	ret
  401149:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401150 <__do_global_dtors_aux>:
  401150:	f3 0f 1e fa          	endbr64
  401154:	80 3d 0d 2f 00 00 00 	cmpb   $0x0,0x2f0d(%rip)        # 404068 <completed.0>
  40115b:	75 2b                	jne    401188 <__do_global_dtors_aux+0x38>
  40115d:	55                   	push   %rbp
  40115e:	48 83 3d 7a 2e 00 00 	cmpq   $0x0,0x2e7a(%rip)        # 403fe0 <__cxa_finalize@GLIBC_2.2.5>
  401165:	00 
  401166:	48 89 e5             	mov    %rsp,%rbp
  401169:	74 0c                	je     401177 <__do_global_dtors_aux+0x27>
  40116b:	48 8b 3d ce 2e 00 00 	mov    0x2ece(%rip),%rdi        # 404040 <__dso_handle>
  401172:	e8 29 ff ff ff       	call   4010a0 <__cxa_finalize@plt>
  401177:	e8 64 ff ff ff       	call   4010e0 <deregister_tm_clones>
  40117c:	c6 05 e5 2e 00 00 01 	movb   $0x1,0x2ee5(%rip)        # 404068 <completed.0>
  401183:	5d                   	pop    %rbp
  401184:	c3                   	ret
  401185:	0f 1f 00             	nopl   (%rax)
  401188:	c3                   	ret
  401189:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401190 <frame_dummy>:
  401190:	f3 0f 1e fa          	endbr64
  401194:	e9 77 ff ff ff       	jmp    401110 <register_tm_clones>
  401199:	0f 1f 00             	nopl   (%rax)

000000000040119c <main>:
  40119c:	55                   	push   %rbp
  40119d:	48 89 e5             	mov    %rsp,%rbp
  4011a0:	48 81 ec b0 00 00 00 	sub    $0xb0,%rsp
  4011a7:	48 89 5d f0          	mov    %rbx,-0x10(%rbp)
  4011ab:	89 7d 80             	mov    %edi,-0x80(%rbp)
  4011ae:	48 89 75 88          	mov    %rsi,-0x78(%rbp)
  4011b2:	b8 00 c2 eb 0b       	mov    $0xbebc200,%eax
  4011b7:	48 89 c7             	mov    %rax,%rdi
  4011ba:	e8 b1 fe ff ff       	call   401070 <malloc@plt>
  4011bf:	48 89 45 90          	mov    %rax,-0x70(%rbp)
  4011c3:	48 8b 45 90          	mov    -0x70(%rbp),%rax
  4011c7:	48 89 45 98          	mov    %rax,-0x68(%rbp)
  4011cb:	b8 00 c2 eb 0b       	mov    $0xbebc200,%eax
  4011d0:	48 89 c7             	mov    %rax,%rdi
  4011d3:	e8 98 fe ff ff       	call   401070 <malloc@plt>
  4011d8:	48 89 45 a0          	mov    %rax,-0x60(%rbp)
  4011dc:	48 8b 45 a0          	mov    -0x60(%rbp),%rax
  4011e0:	48 89 45 a8          	mov    %rax,-0x58(%rbp)
  4011e4:	b8 00 c2 eb 0b       	mov    $0xbebc200,%eax
  4011e9:	48 89 c7             	mov    %rax,%rdi
  4011ec:	e8 7f fe ff ff       	call   401070 <malloc@plt>
  4011f1:	48 89 45 b0          	mov    %rax,-0x50(%rbp)
  4011f5:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
  4011f9:	48 89 45 b8          	mov    %rax,-0x48(%rbp)
  4011fd:	c7 85 50 ff ff ff 00 	movl   $0x0,-0xb0(%rbp)
  401204:	00 00 00 
  401207:	8b 85 50 ff ff ff    	mov    -0xb0(%rbp),%eax
  40120d:	3d 88 13 00 00       	cmp    $0x1388,%eax
  401212:	7c 18                	jl     40122c <main+0x90>
  401214:	e9 cc 00 00 00       	jmp    4012e5 <main+0x149>
  401219:	b8 01 00 00 00       	mov    $0x1,%eax
  40121e:	03 85 50 ff ff ff    	add    -0xb0(%rbp),%eax
  401224:	89 85 50 ff ff ff    	mov    %eax,-0xb0(%rbp)
  40122a:	eb db                	jmp    401207 <main+0x6b>
  40122c:	c7 85 54 ff ff ff 00 	movl   $0x0,-0xac(%rbp)
  401233:	00 00 00 
  401236:	8b 85 54 ff ff ff    	mov    -0xac(%rbp),%eax
  40123c:	3d 88 13 00 00       	cmp    $0x1388,%eax
  401241:	7d d6                	jge    401219 <main+0x7d>
  401243:	f2 0f 10 05 bd 0d 00 	movsd  0xdbd(%rip),%xmm0        # 402008 <_IO_stdin_used+0x8>
  40124a:	00 
  40124b:	8b 85 50 ff ff ff    	mov    -0xb0(%rbp),%eax
  401251:	48 63 c0             	movslq %eax,%rax
  401254:	48 69 c0 40 9c 00 00 	imul   $0x9c40,%rax,%rax
  40125b:	48 03 45 98          	add    -0x68(%rbp),%rax
  40125f:	8b 95 54 ff ff ff    	mov    -0xac(%rbp),%edx
  401265:	48 63 d2             	movslq %edx,%rdx
  401268:	48 6b d2 08          	imul   $0x8,%rdx,%rdx
  40126c:	48 03 c2             	add    %rdx,%rax
  40126f:	f2 0f 11 00          	movsd  %xmm0,(%rax)
  401273:	f2 0f 10 05 95 0d 00 	movsd  0xd95(%rip),%xmm0        # 402010 <_IO_stdin_used+0x10>
  40127a:	00 
  40127b:	8b 85 50 ff ff ff    	mov    -0xb0(%rbp),%eax
  401281:	48 63 c0             	movslq %eax,%rax
  401284:	48 69 c0 40 9c 00 00 	imul   $0x9c40,%rax,%rax
  40128b:	48 03 45 a8          	add    -0x58(%rbp),%rax
  40128f:	8b 95 54 ff ff ff    	mov    -0xac(%rbp),%edx
  401295:	48 63 d2             	movslq %edx,%rdx
  401298:	48 6b d2 08          	imul   $0x8,%rdx,%rdx
  40129c:	48 03 c2             	add    %rdx,%rax
  40129f:	f2 0f 11 00          	movsd  %xmm0,(%rax)
  4012a3:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4012a7:	8b 85 50 ff ff ff    	mov    -0xb0(%rbp),%eax
  4012ad:	48 63 c0             	movslq %eax,%rax
  4012b0:	48 69 c0 40 9c 00 00 	imul   $0x9c40,%rax,%rax
  4012b7:	48 03 45 b8          	add    -0x48(%rbp),%rax
  4012bb:	8b 95 54 ff ff ff    	mov    -0xac(%rbp),%edx
  4012c1:	48 63 d2             	movslq %edx,%rdx
  4012c4:	48 6b d2 08          	imul   $0x8,%rdx,%rdx
  4012c8:	48 03 c2             	add    %rdx,%rax
  4012cb:	f2 0f 11 00          	movsd  %xmm0,(%rax)
  4012cf:	b8 01 00 00 00       	mov    $0x1,%eax
  4012d4:	03 85 54 ff ff ff    	add    -0xac(%rbp),%eax
  4012da:	89 85 54 ff ff ff    	mov    %eax,-0xac(%rbp)
  4012e0:	e9 51 ff ff ff       	jmp    401236 <main+0x9a>
  4012e5:	b8 00 00 00 00       	mov    $0x0,%eax
  4012ea:	83 f8 01             	cmp    $0x1,%eax
  4012ed:	74 11                	je     401300 <main+0x164>
  4012ef:	e8 4c fd ff ff       	call   401040 <clock@plt>
  4012f4:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
  4012f8:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  4012fc:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  401300:	c7 85 58 ff ff ff 00 	movl   $0x0,-0xa8(%rbp)
  401307:	00 00 00 
  40130a:	8b 85 58 ff ff ff    	mov    -0xa8(%rbp),%eax
  401310:	3d 88 13 00 00       	cmp    $0x1388,%eax
  401315:	7c 18                	jl     40132f <main+0x193>
  401317:	e9 14 01 00 00       	jmp    401430 <main+0x294>
  40131c:	b8 01 00 00 00       	mov    $0x1,%eax
  401321:	03 85 58 ff ff ff    	add    -0xa8(%rbp),%eax
  401327:	89 85 58 ff ff ff    	mov    %eax,-0xa8(%rbp)
  40132d:	eb db                	jmp    40130a <main+0x16e>
  40132f:	c7 85 5c ff ff ff 00 	movl   $0x0,-0xa4(%rbp)
  401336:	00 00 00 
  401339:	8b 85 5c ff ff ff    	mov    -0xa4(%rbp),%eax
  40133f:	3d 88 13 00 00       	cmp    $0x1388,%eax
  401344:	7c 15                	jl     40135b <main+0x1bf>
  401346:	eb d4                	jmp    40131c <main+0x180>
  401348:	b8 01 00 00 00       	mov    $0x1,%eax
  40134d:	03 85 5c ff ff ff    	add    -0xa4(%rbp),%eax
  401353:	89 85 5c ff ff ff    	mov    %eax,-0xa4(%rbp)
  401359:	eb de                	jmp    401339 <main+0x19d>
  40135b:	c7 85 60 ff ff ff 00 	movl   $0x0,-0xa0(%rbp)
  401362:	00 00 00 
  401365:	8b 85 60 ff ff ff    	mov    -0xa0(%rbp),%eax
  40136b:	3d 88 13 00 00       	cmp    $0x1388,%eax
  401370:	7d d6                	jge    401348 <main+0x1ac>
  401372:	8b 85 58 ff ff ff    	mov    -0xa8(%rbp),%eax
  401378:	48 63 c0             	movslq %eax,%rax
  40137b:	48 69 c0 40 9c 00 00 	imul   $0x9c40,%rax,%rax
  401382:	48 03 45 b8          	add    -0x48(%rbp),%rax
  401386:	8b 95 60 ff ff ff    	mov    -0xa0(%rbp),%edx
  40138c:	48 63 d2             	movslq %edx,%rdx
  40138f:	48 6b d2 08          	imul   $0x8,%rdx,%rdx
  401393:	48 03 c2             	add    %rdx,%rax
  401396:	8b 95 58 ff ff ff    	mov    -0xa8(%rbp),%edx
  40139c:	48 63 d2             	movslq %edx,%rdx
  40139f:	48 69 d2 40 9c 00 00 	imul   $0x9c40,%rdx,%rdx
  4013a6:	48 03 55 98          	add    -0x68(%rbp),%rdx
  4013aa:	8b 8d 5c ff ff ff    	mov    -0xa4(%rbp),%ecx
  4013b0:	48 63 c9             	movslq %ecx,%rcx
  4013b3:	48 6b c9 08          	imul   $0x8,%rcx,%rcx
  4013b7:	48 03 d1             	add    %rcx,%rdx
  4013ba:	f2 0f 10 02          	movsd  (%rdx),%xmm0
  4013be:	8b 95 5c ff ff ff    	mov    -0xa4(%rbp),%edx
  4013c4:	48 63 d2             	movslq %edx,%rdx
  4013c7:	48 69 d2 40 9c 00 00 	imul   $0x9c40,%rdx,%rdx
  4013ce:	48 03 55 a8          	add    -0x58(%rbp),%rdx
  4013d2:	8b 8d 60 ff ff ff    	mov    -0xa0(%rbp),%ecx
  4013d8:	48 63 c9             	movslq %ecx,%rcx
  4013db:	48 6b c9 08          	imul   $0x8,%rcx,%rcx
  4013df:	48 03 d1             	add    %rcx,%rdx
  4013e2:	f2 0f 10 0a          	movsd  (%rdx),%xmm1
  4013e6:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  4013ea:	f2 0f 10 08          	movsd  (%rax),%xmm1
  4013ee:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4013f2:	8b 85 58 ff ff ff    	mov    -0xa8(%rbp),%eax
  4013f8:	48 63 c0             	movslq %eax,%rax
  4013fb:	48 69 c0 40 9c 00 00 	imul   $0x9c40,%rax,%rax
  401402:	48 03 45 b8          	add    -0x48(%rbp),%rax
  401406:	8b 95 60 ff ff ff    	mov    -0xa0(%rbp),%edx
  40140c:	48 63 d2             	movslq %edx,%rdx
  40140f:	48 6b d2 08          	imul   $0x8,%rdx,%rdx
  401413:	48 03 c2             	add    %rdx,%rax
  401416:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  40141a:	b8 01 00 00 00       	mov    $0x1,%eax
  40141f:	03 85 60 ff ff ff    	add    -0xa0(%rbp),%eax
  401425:	89 85 60 ff ff ff    	mov    %eax,-0xa0(%rbp)
  40142b:	e9 35 ff ff ff       	jmp    401365 <main+0x1c9>
  401430:	b8 00 00 00 00       	mov    $0x0,%eax
  401435:	83 f8 01             	cmp    $0x1,%eax
  401438:	74 60                	je     40149a <main+0x2fe>
  40143a:	e8 01 fc ff ff       	call   401040 <clock@plt>
  40143f:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
  401443:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  401447:	48 f7 d8             	neg    %rax
  40144a:	48 03 45 d0          	add    -0x30(%rbp),%rax
  40144e:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  401452:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  401456:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
  40145b:	f2 0f 10 0d b5 0b 00 	movsd  0xbb5(%rip),%xmm1        # 402018 <_IO_stdin_used+0x18>
  401462:	00 
  401463:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  401467:	f2 0f 11 45 e0       	movsd  %xmm0,-0x20(%rbp)
  40146c:	48 8b 05 ed 2b 00 00 	mov    0x2bed(%rip),%rax        # 404060 <stderr@GLIBC_2.2.5>
  401473:	ba 20 20 40 00       	mov    $0x402020,%edx
  401478:	b9 88 13 00 00       	mov    $0x1388,%ecx
  40147d:	f2 0f 10 45 e0       	movsd  -0x20(%rbp),%xmm0
  401482:	48 89 c7             	mov    %rax,%rdi
  401485:	48 89 d6             	mov    %rdx,%rsi
  401488:	89 ca                	mov    %ecx,%edx
  40148a:	b8 01 00 00 00       	mov    $0x1,%eax
  40148f:	e8 cc fb ff ff       	call   401060 <fprintf@plt>
  401494:	89 85 64 ff ff ff    	mov    %eax,-0x9c(%rbp)
  40149a:	b8 3c 20 40 00       	mov    $0x40203c,%eax
  40149f:	ba 48 20 40 00       	mov    $0x402048,%edx
  4014a4:	48 89 c7             	mov    %rax,%rdi
  4014a7:	48 89 d6             	mov    %rdx,%rsi
  4014aa:	e8 d1 fb ff ff       	call   401080 <fopen@plt>
  4014af:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
  4014b3:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4014b7:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4014bb:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4014bf:	48 85 c0             	test   %rax,%rax
  4014c2:	75 18                	jne    4014dc <main+0x340>
  4014c4:	b8 4c 20 40 00       	mov    $0x40204c,%eax
  4014c9:	48 89 c7             	mov    %rax,%rdi
  4014cc:	e8 bf fb ff ff       	call   401090 <perror@plt>
  4014d1:	b8 01 00 00 00       	mov    $0x1,%eax
  4014d6:	48 8b 5d f0          	mov    -0x10(%rbp),%rbx
  4014da:	c9                   	leave
  4014db:	c3                   	ret
  4014dc:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4014e0:	ba 54 20 40 00       	mov    $0x402054,%edx
  4014e5:	b9 88 13 00 00       	mov    $0x1388,%ecx
  4014ea:	48 89 c7             	mov    %rax,%rdi
  4014ed:	48 89 d6             	mov    %rdx,%rsi
  4014f0:	89 ca                	mov    %ecx,%edx
  4014f2:	b8 00 00 00 00       	mov    $0x0,%eax
  4014f7:	e8 64 fb ff ff       	call   401060 <fprintf@plt>
  4014fc:	89 85 68 ff ff ff    	mov    %eax,-0x98(%rbp)
  401502:	c7 85 6c ff ff ff 00 	movl   $0x0,-0x94(%rbp)
  401509:	00 00 00 
  40150c:	8b 85 6c ff ff ff    	mov    -0x94(%rbp),%eax
  401512:	3d e8 03 00 00       	cmp    $0x3e8,%eax
  401517:	7c 18                	jl     401531 <main+0x395>
  401519:	e9 ac 00 00 00       	jmp    4015ca <main+0x42e>
  40151e:	b8 01 00 00 00       	mov    $0x1,%eax
  401523:	03 85 6c ff ff ff    	add    -0x94(%rbp),%eax
  401529:	89 85 6c ff ff ff    	mov    %eax,-0x94(%rbp)
  40152f:	eb db                	jmp    40150c <main+0x370>
  401531:	c7 85 70 ff ff ff 00 	movl   $0x0,-0x90(%rbp)
  401538:	00 00 00 
  40153b:	8b 85 70 ff ff ff    	mov    -0x90(%rbp),%eax
  401541:	3d e8 03 00 00       	cmp    $0x3e8,%eax
  401546:	7c 15                	jl     40155d <main+0x3c1>
  401548:	eb 5c                	jmp    4015a6 <main+0x40a>
  40154a:	b8 01 00 00 00       	mov    $0x1,%eax
  40154f:	03 85 70 ff ff ff    	add    -0x90(%rbp),%eax
  401555:	89 85 70 ff ff ff    	mov    %eax,-0x90(%rbp)
  40155b:	eb de                	jmp    40153b <main+0x39f>
  40155d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  401561:	ba 5c 20 40 00       	mov    $0x40205c,%edx
  401566:	8b 8d 6c ff ff ff    	mov    -0x94(%rbp),%ecx
  40156c:	48 63 c9             	movslq %ecx,%rcx
  40156f:	48 69 c9 40 9c 00 00 	imul   $0x9c40,%rcx,%rcx
  401576:	48 03 4d b8          	add    -0x48(%rbp),%rcx
  40157a:	8b 9d 70 ff ff ff    	mov    -0x90(%rbp),%ebx
  401580:	48 63 db             	movslq %ebx,%rbx
  401583:	48 6b db 08          	imul   $0x8,%rbx,%rbx
  401587:	48 03 cb             	add    %rbx,%rcx
  40158a:	f2 0f 10 01          	movsd  (%rcx),%xmm0
  40158e:	48 89 c7             	mov    %rax,%rdi
  401591:	48 89 d6             	mov    %rdx,%rsi
  401594:	b8 01 00 00 00       	mov    $0x1,%eax
  401599:	e8 c2 fa ff ff       	call   401060 <fprintf@plt>
  40159e:	89 85 78 ff ff ff    	mov    %eax,-0x88(%rbp)
  4015a4:	eb a4                	jmp    40154a <main+0x3ae>
  4015a6:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4015aa:	ba 64 20 40 00       	mov    $0x402064,%edx
  4015af:	48 89 c7             	mov    %rax,%rdi
  4015b2:	48 89 d6             	mov    %rdx,%rsi
  4015b5:	b8 00 00 00 00       	mov    $0x0,%eax
  4015ba:	e8 a1 fa ff ff       	call   401060 <fprintf@plt>
  4015bf:	89 85 7c ff ff ff    	mov    %eax,-0x84(%rbp)
  4015c5:	e9 54 ff ff ff       	jmp    40151e <main+0x382>
  4015ca:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4015ce:	48 89 c7             	mov    %rax,%rdi
  4015d1:	e8 7a fa ff ff       	call   401050 <fclose@plt>
  4015d6:	89 85 74 ff ff ff    	mov    %eax,-0x8c(%rbp)
  4015dc:	48 8b 45 98          	mov    -0x68(%rbp),%rax
  4015e0:	48 89 c7             	mov    %rax,%rdi
  4015e3:	e8 48 fa ff ff       	call   401030 <free@plt>
  4015e8:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
  4015ec:	48 89 c7             	mov    %rax,%rdi
  4015ef:	e8 3c fa ff ff       	call   401030 <free@plt>
  4015f4:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  4015f8:	48 89 c7             	mov    %rax,%rdi
  4015fb:	e8 30 fa ff ff       	call   401030 <free@plt>
  401600:	b8 00 00 00 00       	mov    $0x0,%eax
  401605:	48 8b 5d f0          	mov    -0x10(%rbp),%rbx
  401609:	c9                   	leave
  40160a:	c3                   	ret
  40160b:	90                   	nop

Disassembly of section .fini:

000000000040160c <_fini>:
  40160c:	48 83 ec 08          	sub    $0x8,%rsp
  401610:	48 83 c4 08          	add    $0x8,%rsp
  401614:	c3                   	ret
