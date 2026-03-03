import { cn } from '../../lib/utils';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'outline';
}

export function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium transition-colors',
        variant === 'default' && 'bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))]',
        variant === 'outline' && 'border text-[hsl(var(--foreground))]',
        className,
      )}
      {...props}
    />
  );
}
