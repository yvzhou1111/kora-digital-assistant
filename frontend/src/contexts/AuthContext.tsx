import React, { createContext, useState, useEffect, ReactNode } from 'react';
import { User, UserRole } from '@/types/user';

interface AuthContextType {
  isAuthenticated: boolean;
  user: User | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (userData: Partial<User>, password: string) => Promise<void>;
  logout: () => void;
  error: string | null;
}

const defaultValue: AuthContextType = {
  isAuthenticated: false,
  user: null,
  isLoading: true,
  login: async () => {},
  register: async () => {},
  logout: () => {},
  error: null,
};

export const AuthContext = createContext<AuthContextType>(defaultValue);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Check if user is already logged in
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        // In a real app, we would check the token validity with the backend
        const token = localStorage.getItem('token');
        
        if (token) {
          // Mock user data - would be fetched from backend in a real app
          const userData: User = JSON.parse(localStorage.getItem('user') || '{}');
          
          // Convert string dates back to Date objects
          if (userData.createdAt) userData.createdAt = new Date(userData.createdAt);
          if (userData.updatedAt) userData.updatedAt = new Date(userData.updatedAt);
          
          setUser(userData);
          setIsAuthenticated(true);
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        // Clear potentially corrupted auth data
        localStorage.removeItem('token');
        localStorage.removeItem('user');
      } finally {
        setIsLoading(false);
      }
    };
    
    checkAuthStatus();
  }, []);
  
  const login = async (email: string, password: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Mock API call - would be a real API call in a real app
      // This is for demonstration purposes only
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock successful login
      if (email === 'test@example.com' && password === 'password') {
        const now = new Date();
        const mockUser: User = {
          id: '1',
          email: 'test@example.com',
          firstName: '测试',
          lastName: '用户',
          displayName: '测试用户',
          avatar: '/images/avatar.png',
          role: UserRole.INDIVIDUAL,
          createdAt: now,
          updatedAt: now,
        };
        
        localStorage.setItem('token', 'mock-token-12345');
        localStorage.setItem('user', JSON.stringify(mockUser));
        
        setUser(mockUser);
        setIsAuthenticated(true);
      } else {
        throw new Error('用户名或密码错误');
      }
    } catch (err) {
      setError((err as Error).message);
      setIsAuthenticated(false);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };
  
  const register = async (userData: Partial<User>, password: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Mock API call - would be a real API call in a real app
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock successful registration
      const now = new Date();
      const mockUser: User = {
        id: Math.random().toString(36).substr(2, 9),
        email: userData.email || 'user@example.com',
        firstName: userData.firstName || '新',
        lastName: userData.lastName || '用户',
        displayName: userData.displayName || '新用户',
        avatar: userData.avatar || '/images/default-avatar.png',
        role: userData.role || UserRole.INDIVIDUAL,
        createdAt: now,
        updatedAt: now,
      };
      
      localStorage.setItem('token', 'mock-token-' + Math.random().toString(36).substr(2, 9));
      localStorage.setItem('user', JSON.stringify(mockUser));
      
      setUser(mockUser);
      setIsAuthenticated(true);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };
  
  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setIsAuthenticated(false);
    setUser(null);
  };
  
  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        user,
        isLoading,
        login,
        register,
        logout,
        error,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}; 