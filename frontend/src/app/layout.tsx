import { Metadata } from 'next';
import '../styles/globals.scss';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { AuthProvider } from '@/contexts/AuthContext';
import { UserProvider } from '@/contexts/UserContext';
import { TwinProvider } from '@/contexts/TwinContext';

export const metadata: Metadata = {
  title: '数字协同人网页',
  description: '赋能个体，连接未来的数字分身平台',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body>
        <ConfigProvider locale={zhCN}>
          <AuthProvider>
            <UserProvider>
              <TwinProvider>
                {children}
              </TwinProvider>
            </UserProvider>
          </AuthProvider>
        </ConfigProvider>
      </body>
    </html>
  );
} 