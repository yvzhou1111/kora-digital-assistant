import React, { createContext, useState, useEffect, ReactNode, useContext } from 'react';
import { AuthContext } from './AuthContext';
import { 
  DigitalTwin, 
  TwinStatus, 
  TrainingStatus, 
  CommunicationStyle,
  TwinConversation,
  KnowledgeSource,
  TwinSkill,
  KnowledgeSourceType,
  KnowledgeSourceStatus
} from '@/types/twin';

interface TwinContextType {
  twins: DigitalTwin[];
  currentTwin: DigitalTwin | null;
  conversations: TwinConversation[];
  currentConversation: TwinConversation | null;
  knowledgeSources: KnowledgeSource[];
  skills: TwinSkill[];
  isLoading: boolean;
  error: string | null;
  createTwin: (data: Partial<DigitalTwin>) => Promise<DigitalTwin>;
  updateTwin: (id: string, data: Partial<DigitalTwin>) => Promise<DigitalTwin>;
  deleteTwin: (id: string) => Promise<void>;
  setCurrentTwin: (twin: DigitalTwin | null) => void;
  createConversation: (twinId: string, title: string) => Promise<TwinConversation>;
  sendMessage: (conversationId: string, content: string) => Promise<void>;
  addKnowledgeSource: (twinId: string, source: Partial<KnowledgeSource>) => Promise<KnowledgeSource>;
}

const defaultValue: TwinContextType = {
  twins: [],
  currentTwin: null,
  conversations: [],
  currentConversation: null,
  knowledgeSources: [],
  skills: [],
  isLoading: false,
  error: null,
  createTwin: async () => ({ id: '' } as DigitalTwin),
  updateTwin: async () => ({ id: '' } as DigitalTwin),
  deleteTwin: async () => {},
  setCurrentTwin: () => {},
  createConversation: async () => ({ id: '' } as TwinConversation),
  sendMessage: async () => {},
  addKnowledgeSource: async () => ({ id: '' } as KnowledgeSource),
};

export const TwinContext = createContext<TwinContextType>(defaultValue);

interface TwinProviderProps {
  children: ReactNode;
}

export const TwinProvider: React.FC<TwinProviderProps> = ({ children }) => {
  const { user, isAuthenticated } = useContext(AuthContext);
  const [twins, setTwins] = useState<DigitalTwin[]>([]);
  const [currentTwin, setCurrentTwin] = useState<DigitalTwin | null>(null);
  const [conversations, setConversations] = useState<TwinConversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<TwinConversation | null>(null);
  const [knowledgeSources, setKnowledgeSources] = useState<KnowledgeSource[]>([]);
  const [skills, setSkills] = useState<TwinSkill[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch user's twins when authenticated
  useEffect(() => {
    const fetchTwins = async () => {
      if (!isAuthenticated || !user) {
        setTwins([]);
        setCurrentTwin(null);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        // In a real app, we would fetch twins from the backend
        await new Promise(resolve => setTimeout(resolve, 800));

        // Create mock twins
        const mockTwins: DigitalTwin[] = [
          {
            id: '1',
            userId: user.id,
            name: '职业分身',
            description: '我的职业数字分身，适用于求职场景',
            avatar: '/images/twin1.png',
            createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
            updatedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
            isPublic: true,
            status: TwinStatus.ACTIVE,
            trainingStatus: TrainingStatus.COMPLETED,
            communicationStyle: CommunicationStyle.PROFESSIONAL,
            knowledgeSourceIds: ['1', '2', '3'],
            skillLevel: 4,
            configurations: {
              responseLength: 'medium',
              knowledgeDepth: 'expert',
              automaticReplies: true,
              aiModel: 'gpt-4',
            },
          },
          {
            id: '2',
            userId: user.id,
            name: '咨询专家分身',
            description: '我的专业知识数字分身，用于分享专业知识',
            avatar: '/images/twin2.png',
            createdAt: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000), // 15 days ago
            updatedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000), // 1 day ago
            isPublic: true,
            status: TwinStatus.ACTIVE,
            trainingStatus: TrainingStatus.COMPLETED,
            communicationStyle: CommunicationStyle.DETAILED,
            knowledgeSourceIds: ['4', '5'],
            skillLevel: 5,
            configurations: {
              responseLength: 'long',
              knowledgeDepth: 'expert',
              automaticReplies: true,
              aiModel: 'gpt-4',
            },
          },
        ];

        setTwins(mockTwins);
        setCurrentTwin(mockTwins[0]);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTwins();
  }, [user, isAuthenticated]);

  // Fetch conversations when currentTwin changes
  useEffect(() => {
    const fetchConversations = async () => {
      if (!currentTwin) {
        setConversations([]);
        setCurrentConversation(null);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        // In a real app, we would fetch conversations from the backend
        await new Promise(resolve => setTimeout(resolve, 500));

        // Create mock conversations
        const mockConversations: TwinConversation[] = [
          {
            id: '1',
            twinId: currentTwin.id,
            userId: user?.id || '',
            title: '关于项目经验的对话',
            createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
            updatedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
            lastMessageAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
            messages: [
              {
                id: '1-1',
                conversationId: '1',
                content: '你能介绍一下你的项目经验吗？',
                sender: 'user',
                timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
                status: 'read',
              },
              {
                id: '1-2',
                conversationId: '1',
                content: '我有超过5年的项目经验，主要专注于前端开发和用户体验设计。我参与过多个大型项目，包括电商平台、企业管理系统和社交媒体应用。',
                sender: 'twin',
                timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000 + 1000),
                status: 'read',
              },
            ],
          },
          {
            id: '2',
            twinId: currentTwin.id,
            userId: user?.id || '',
            title: '技术栈讨论',
            createdAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
            updatedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
            lastMessageAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
            messages: [
              {
                id: '2-1',
                conversationId: '2',
                content: '你熟悉哪些前端技术栈？',
                sender: 'user',
                timestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
                status: 'read',
              },
              {
                id: '2-2',
                conversationId: '2',
                content: '我精通React、Vue和Angular框架，以及TypeScript、Next.js、Webpack等工具。我还有使用Redux和Vuex进行状态管理的丰富经验。',
                sender: 'twin',
                timestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000 + 1000),
                status: 'read',
              },
            ],
          },
        ];

        setConversations(mockConversations);
        setCurrentConversation(mockConversations[0]);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchConversations();
  }, [currentTwin, user]);

  // Fetch knowledge sources and skills when currentTwin changes
  useEffect(() => {
    const fetchTwinData = async () => {
      if (!currentTwin) {
        setKnowledgeSources([]);
        setSkills([]);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        // In a real app, we would fetch data from the backend
        await new Promise(resolve => setTimeout(resolve, 600));

        // Create mock knowledge sources
        const mockKnowledgeSources: KnowledgeSource[] = [
          {
            id: '1',
            twinId: currentTwin.id,
            type: KnowledgeSourceType.RESUME,
            name: '个人简历',
            fileUrl: '/documents/resume.pdf',
            status: KnowledgeSourceStatus.COMPLETED,
            createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
            updatedAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
          },
          {
            id: '2',
            twinId: currentTwin.id,
            type: KnowledgeSourceType.LINKEDIN,
            name: 'LinkedIn资料',
            url: 'https://linkedin.com/in/username',
            status: KnowledgeSourceStatus.COMPLETED,
            createdAt: new Date(Date.now() - 25 * 24 * 60 * 60 * 1000),
            updatedAt: new Date(Date.now() - 25 * 24 * 60 * 60 * 1000),
          },
          {
            id: '3',
            twinId: currentTwin.id,
            type: KnowledgeSourceType.MANUAL_ENTRY,
            name: '补充技能',
            content: '前端开发、React、TypeScript、用户体验设计',
            status: KnowledgeSourceStatus.COMPLETED,
            createdAt: new Date(Date.now() - 20 * 24 * 60 * 60 * 1000),
            updatedAt: new Date(Date.now() - 20 * 24 * 60 * 60 * 1000),
          },
        ];

        // Create mock skills
        const mockSkills: TwinSkill[] = [
          {
            id: '1',
            twinId: currentTwin.id,
            name: 'React',
            category: '前端开发',
            level: 5,
            confidence: 95,
            sources: ['1', '2', '3'],
          },
          {
            id: '2',
            twinId: currentTwin.id,
            name: 'TypeScript',
            category: '编程语言',
            level: 4,
            confidence: 90,
            sources: ['1', '3'],
          },
          {
            id: '3',
            twinId: currentTwin.id,
            name: '用户体验设计',
            category: '设计',
            level: 4,
            confidence: 85,
            sources: ['1', '3'],
          },
          {
            id: '4',
            twinId: currentTwin.id,
            name: '项目管理',
            category: '管理',
            level: 3,
            confidence: 75,
            sources: ['1', '2'],
          },
        ];

        setKnowledgeSources(mockKnowledgeSources);
        setSkills(mockSkills);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTwinData();
  }, [currentTwin]);

  // Create a new digital twin
  const createTwin = async (data: Partial<DigitalTwin>): Promise<DigitalTwin> => {
    setIsLoading(true);
    setError(null);

    try {
      // In a real app, we would send the twin data to the backend
      await new Promise(resolve => setTimeout(resolve, 1000));

      const now = new Date();
      const newTwin: DigitalTwin = {
        id: Math.random().toString(36).substr(2, 9),
        userId: user?.id || '',
        name: data.name || '新数字分身',
        description: data.description || '',
        avatar: data.avatar || '/images/default-twin.png',
        createdAt: now,
        updatedAt: now,
        isPublic: data.isPublic || false,
        status: TwinStatus.PENDING,
        trainingStatus: TrainingStatus.NOT_STARTED,
        communicationStyle: data.communicationStyle || CommunicationStyle.PROFESSIONAL,
        knowledgeSourceIds: data.knowledgeSourceIds || [],
        skillLevel: data.skillLevel || 1,
        configurations: data.configurations || {
          responseLength: 'medium',
          knowledgeDepth: 'intermediate',
          automaticReplies: false,
          aiModel: 'gpt-3.5-turbo',
        },
      };

      setTwins(prev => [...prev, newTwin]);
      setCurrentTwin(newTwin);

      return newTwin;
    } catch (err) {
      setError((err as Error).message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Update an existing digital twin
  const updateTwin = async (id: string, data: Partial<DigitalTwin>): Promise<DigitalTwin> => {
    setIsLoading(true);
    setError(null);

    try {
      // In a real app, we would send the updated data to the backend
      await new Promise(resolve => setTimeout(resolve, 800));

      const updatedTwins = twins.map(twin => {
        if (twin.id === id) {
          const updatedTwin = {
            ...twin,
            ...data,
            updatedAt: new Date(),
          };
          
          // If this is the current twin, update it as well
          if (currentTwin && currentTwin.id === id) {
            setCurrentTwin(updatedTwin);
          }
          
          return updatedTwin;
        }
        return twin;
      });

      setTwins(updatedTwins);
      const updatedTwin = updatedTwins.find(twin => twin.id === id);
      
      if (!updatedTwin) {
        throw new Error('未找到数字分身');
      }
      
      return updatedTwin;
    } catch (err) {
      setError((err as Error).message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Delete a digital twin
  const deleteTwin = async (id: string): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      // In a real app, we would send a delete request to the backend
      await new Promise(resolve => setTimeout(resolve, 600));

      const updatedTwins = twins.filter(twin => twin.id !== id);
      setTwins(updatedTwins);
      
      // If we're deleting the current twin, reset it
      if (currentTwin && currentTwin.id === id) {
        setCurrentTwin(updatedTwins.length > 0 ? updatedTwins[0] : null);
      }
    } catch (err) {
      setError((err as Error).message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Create a new conversation
  const createConversation = async (twinId: string, title: string): Promise<TwinConversation> => {
    setIsLoading(true);
    setError(null);

    try {
      // In a real app, we would send the conversation data to the backend
      await new Promise(resolve => setTimeout(resolve, 500));

      const now = new Date();
      const newConversation: TwinConversation = {
        id: Math.random().toString(36).substr(2, 9),
        twinId,
        userId: user?.id || '',
        title,
        createdAt: now,
        updatedAt: now,
        lastMessageAt: now,
        messages: [],
      };

      setConversations(prev => [...prev, newConversation]);
      setCurrentConversation(newConversation);

      return newConversation;
    } catch (err) {
      setError((err as Error).message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Send a message in a conversation
  const sendMessage = async (conversationId: string, content: string): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      // In a real app, we would send the message to the backend
      await new Promise(resolve => setTimeout(resolve, 300));

      const now = new Date();
      const userMessage = {
        id: Math.random().toString(36).substr(2, 9),
        conversationId,
        content,
        sender: 'user' as const,
        timestamp: now,
        status: 'sent' as const,
      };

      // Update the conversations with the new message
      const updatedConversations = conversations.map(conversation => {
        if (conversation.id === conversationId) {
          const updatedConversation = {
            ...conversation,
            messages: [...conversation.messages, userMessage],
            updatedAt: now,
            lastMessageAt: now,
          };
          
          // If this is the current conversation, update it as well
          if (currentConversation && currentConversation.id === conversationId) {
            setCurrentConversation(updatedConversation);
          }
          
          return updatedConversation;
        }
        return conversation;
      });

      setConversations(updatedConversations);

      // Simulate twin response after a delay
      setTimeout(() => {
        const twinResponse = {
          id: Math.random().toString(36).substr(2, 9),
          conversationId,
          content: `这是数字分身的回复: ${content.length > 10 ? '您的问题很有深度' : '请详细描述您的问题'}。我能帮您解决这个问题，请提供更多细节。`,
          sender: 'twin' as const,
          timestamp: new Date(),
          status: 'sent' as const,
        };

        setConversations(prev => {
          const updated = prev.map(conversation => {
            if (conversation.id === conversationId) {
              const updatedConversation = {
                ...conversation,
                messages: [...conversation.messages, twinResponse],
                updatedAt: new Date(),
                lastMessageAt: new Date(),
              };
              
              // If this is the current conversation, update it as well
              if (currentConversation && currentConversation.id === conversationId) {
                setCurrentConversation(updatedConversation);
              }
              
              return updatedConversation;
            }
            return conversation;
          });
          
          return updated;
        });
      }, 1000);

    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  // Add a knowledge source to a twin
  const addKnowledgeSource = async (twinId: string, source: Partial<KnowledgeSource>): Promise<KnowledgeSource> => {
    setIsLoading(true);
    setError(null);

    try {
      // In a real app, we would send the source data to the backend
      await new Promise(resolve => setTimeout(resolve, 700));

      const now = new Date();
      const newSource: KnowledgeSource = {
        id: Math.random().toString(36).substr(2, 9),
        twinId,
        type: source.type || KnowledgeSourceType.MANUAL_ENTRY,
        name: source.name || '新知识源',
        content: source.content,
        url: source.url,
        fileUrl: source.fileUrl,
        status: KnowledgeSourceStatus.PENDING,
        createdAt: now,
        updatedAt: now,
      };

      setKnowledgeSources(prev => [...prev, newSource]);

      // Update the twin's knowledgeSourceIds
      if (currentTwin && currentTwin.id === twinId) {
        const updatedTwin = {
          ...currentTwin,
          knowledgeSourceIds: [...currentTwin.knowledgeSourceIds, newSource.id],
          updatedAt: now,
        };
        
        setCurrentTwin(updatedTwin);
        
        // Also update in the twins array
        setTwins(prev => prev.map(twin => 
          twin.id === twinId ? updatedTwin : twin
        ));
      }

      // Simulate processing and completion after a delay
      setTimeout(() => {
        setKnowledgeSources(prev => prev.map(s => 
          s.id === newSource.id 
            ? { ...s, status: KnowledgeSourceStatus.PROCESSING }
            : s
        ));

        setTimeout(() => {
          setKnowledgeSources(prev => prev.map(s => 
            s.id === newSource.id 
              ? { ...s, status: KnowledgeSourceStatus.COMPLETED, updatedAt: new Date() }
              : s
          ));
        }, 2000);
      }, 1000);

      return newSource;
    } catch (err) {
      setError((err as Error).message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <TwinContext.Provider
      value={{
        twins,
        currentTwin,
        conversations,
        currentConversation,
        knowledgeSources,
        skills,
        isLoading,
        error,
        createTwin,
        updateTwin,
        deleteTwin,
        setCurrentTwin,
        createConversation,
        sendMessage,
        addKnowledgeSource,
      }}
    >
      {children}
    </TwinContext.Provider>
  );
}; 