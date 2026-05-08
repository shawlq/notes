# 记录Ubuntu使用过程常见问题

1. DNS问题：

A： 见 `sandhill_ubuntu_dns_fail_faq.md`

2. Q：Ubuntu打开Termius时，弹出 Authentication required 对话框，输入登录密码没用：

A： 在终端执行命令： `rm ~/.local/share/keyrings/login.keyring` 后恢复